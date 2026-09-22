"""CPU tests for the licensed-asset fetcher: no network, no credentials, no downloads.

The point of these tests is the failure the stage-7 review called out (R1): an asset is
only real once its *content* says so. So the stub server hands back a login page where a
zip should be, and the fetcher has to refuse it rather than count it as downloaded.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

import pytest

from sim2sense_fall.humans.fetch import (
    CredentialError,
    FetchError,
    RemoteAsset,
    Session,
    asset_roots,
    credentials_for_site,
    discover,
    download_asset,
    load_sites,
    parse_page,
    provenance_record,
    read_credentials_file,
    select_assets,
    unpack_archive,
    validate_destination,
    verify_container,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs" / "humans" / "assets.yaml"

PASSWORD = "correct-battery-staple"  # noqa: S105 - a test fixture, never a real secret


# ---------------------------------------------------------------------------
# Registry-derived sites
# ---------------------------------------------------------------------------


def test_load_sites_uses_registry_registration_urls() -> None:
    sites = load_sites(REGISTRY)
    assert set(sites) == {"smpl", "amass"}
    assert sites["smpl"].registration_url == "https://smpl.is.tue.mpg.de/download.php"
    assert sites["smpl"].host == "smpl.is.tue.mpg.de"
    assert sites["smpl"].email_key == "SMPL_EMAIL"
    assert sites["amass"].asset_ids == ("amass",)
    assert "modellicense" in sites["smpl"].license_urls[0]


def test_asset_roots_expands_project_placeholder() -> None:
    roots = asset_roots(REGISTRY, project_root=REPO_ROOT)
    assert roots[0] == REPO_ROOT / "data" / "humans"
    assert all(root.is_absolute() for root in roots)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _write_creds(path: Path, body: str, mode: int = 0o600) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(mode)
    return path


def test_credential_file_rejects_group_readable_mode(tmp_path: Path) -> None:
    path = _write_creds(
        tmp_path / "creds", f"SMPL_EMAIL=a@b.c\nSMPL_PASSWORD={PASSWORD}\n", mode=0o644
    )
    with pytest.raises(CredentialError, match="group/other access"):
        read_credentials_file(path)


def test_credential_file_rejects_malformed_line(tmp_path: Path) -> None:
    path = _write_creds(tmp_path / "creds", "SMPL_EMAIL a@b.c\n")
    with pytest.raises(CredentialError, match="expected KEY=VALUE"):
        read_credentials_file(path)


def test_credential_file_missing_reports_registration_help(tmp_path: Path) -> None:
    with pytest.raises(CredentialError, match="chmod 600"):
        read_credentials_file(tmp_path / "absent")


def test_credentials_for_site_requires_both_halves(tmp_path: Path) -> None:
    site = next(iter(load_sites(REGISTRY).values()))
    path = _write_creds(tmp_path / "creds", "SMPL_EMAIL=only@institution.edu\n")
    values = read_credentials_file(path)
    with pytest.raises(CredentialError) as excinfo:
        credentials_for_site(site, values)
    message = str(excinfo.value)
    assert "smpl.is.tue.mpg.de" in message
    assert "modellicense" in message
    assert PASSWORD not in message


def test_credentials_never_repr_or_str_the_password() -> None:
    credentials = credentials_for_site(
        next(iter(load_sites(REGISTRY).values())),
        {"SMPL_EMAIL": "me@institution.edu", "SMPL_PASSWORD": PASSWORD},
    )
    assert PASSWORD not in repr(credentials)
    assert PASSWORD not in str(credentials)


# ---------------------------------------------------------------------------
# Page parsing: discovery rather than guessed URLs
# ---------------------------------------------------------------------------

SMPL_PAGE = b"""
<html><body>
  <a href="/account/logout/">log out</a>
  <a href="/dataset_license">licence terms</a>
  <a href="https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip">
     SMPL v1.1.0 for Python (female/male/neutral, 300 shape PCs)</a>
  <a href="https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip">
     SMPL v1.0.0 for Python (female/male, 10 shape PCs)</a>
  <a href="/download/uv_map/"><strong>Download UV map in OBJ format</strong></a>
  <form action="/account/login/" method="post">
    <input type="hidden" name="csrfmiddlewaretoken" value="tok">
    <input type="email" name="email" value="">
    <input type="password" name="password">
    <button name="submit" value="1">Sign in</button>
  </form>
</body></html>
"""


def test_parse_page_keeps_asset_links_and_drops_navigation() -> None:
    offer = parse_page(SMPL_PAGE, "https://smpl.is.tue.mpg.de/download.php")
    names = [asset.filename for asset in offer.assets]
    assert names == ["SMPL_python_v.1.1.0.zip", "SMPL_python_v.1.0.0.zip"]
    assert offer.assets[0].label.startswith("SMPL v1.1.0")
    assert offer.assets[0].method == "GET"


def test_parse_page_detects_login_form_and_its_fields() -> None:
    offer = parse_page(SMPL_PAGE, "https://smpl.is.tue.mpg.de/download.php")
    assert offer.asks_password is True
    assert offer.login_form is not None
    assert offer.login_form.action == "https://smpl.is.tue.mpg.de/account/login/"
    assert offer.login_form.email_field == "email"
    assert offer.login_form.password_field == "password"
    assert offer.login_form.fields["csrfmiddlewaretoken"] == "tok"


def test_parse_page_resolves_relative_asset_hrefs() -> None:
    page = b'<a href="/files/AMASS.zip">AMASS</a>'
    offer = parse_page(page, "https://amass.is.tue.mpg.de/download.php")
    assert offer.assets[0].url == "https://amass.is.tue.mpg.de/files/AMASS.zip"
    assert offer.assets[0].filename == "AMASS.zip"


def test_select_assets_matches_label_and_filename_case_insensitively() -> None:
    offer = parse_page(SMPL_PAGE, "https://smpl.is.tue.mpg.de/download.php")
    chosen = select_assets(offer.assets, ("1.0.0",))
    assert [asset.filename for asset in chosen] == ["SMPL_python_v.1.0.0.zip"]
    assert len(select_assets(offer.assets, (), take_all=True)) == 2
    assert select_assets(offer.assets, ("nothing-matches",)) == ()


def test_remote_asset_filename_falls_back_to_url_path_or_label() -> None:
    by_path = RemoteAsset(label="x", url="https://h/a/basicModel_neutral.pkl")
    assert by_path.filename == "basicModel_neutral.pkl"
    by_label = RemoteAsset(label="UV Map OBJ", url="https://h/download.php?thing=3")
    assert by_label.filename == "UV_Map_OBJ"


# ---------------------------------------------------------------------------
# Stub HTTP server: proves the content is checked, not just the status code
# ---------------------------------------------------------------------------

PAYLOAD = b"SMPL-PKIL-BYTES" + bytes(range(256))
#: How much a capped route hands over per connection, mimicking a dropped transfer.
CAP = 120


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return buffer.getvalue()


#: Served at a ``.zip`` URL, so container verification has something real to open.
ZIP_BYTES = _zip_bytes({"basicModel_neutral_lbs_10_207_0_v1.1.0.pkl": b"\x80\x02}"})


class _Stub(BaseHTTPRequestHandler):
    bodies: ClassVar[dict[str, bytes]] = {}
    #: Routes that ignore Range and restart from zero, so a resume can never finish.
    liars: ClassVar[set[str]] = set()
    #: Routes that cut the body short while still declaring the full length.
    caps: ClassVar[set[str]] = set()
    types: ClassVar[dict[str, str]] = {}
    declares: ClassVar[bool] = True

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.split("?")[0]
        whole = type(self).bodies.get(route)
        if whole is None:
            self.send_response(404)
            self.end_headers()
            return
        start = 0
        status = 200
        match = re.fullmatch(r"bytes=(\d+)-", (self.headers.get("Range") or "").strip())
        if match and route not in type(self).liars:
            start = int(match.group(1))
            status = 206 if start else 200
        body = whole[start:]
        if route in type(self).caps:
            body = body[:CAP]
        self.send_response(status)
        self.send_header(
            "Content-Type", type(self).types.get(route, "application/octet-stream")
        )
        if type(self).declares:
            # The honest remaining length, even where the connection is cut short: a
            # truncated body has to look incomplete to the client, as it does in the
            # field, rather than like a finished but short file.
            self.send_header("Content-Length", str(len(whole) - start))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(whole)}")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.do_GET()

    def log_message(self, *args: Any) -> None:  # keep pytest quiet
        pass


@pytest.fixture()
def stub() -> Any:
    _Stub.bodies = {
        "/page": PAYLOAD,
        "/asset.zip": ZIP_BYTES,
        "/capped": PAYLOAD,
        "/liar": PAYLOAD,
        "/refused": b"<!DOCTYPE html>\n<html><body>Please sign in</body></html>",
        "/empty": b"",
    }
    _Stub.liars = {"/liar"}
    _Stub.caps = {"/capped", "/liar"}
    _Stub.types = {"/refused": "text/html"}
    _Stub.declares = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_stream_to_hashes_what_it_writes(tmp_path: Path, stub: str) -> None:
    transfer = Session().stream_to(f"{stub}/page", tmp_path / "asset.zip")
    target = tmp_path / "asset.zip"
    assert transfer.content_type == "application/octet-stream"
    assert (transfer.bytes, transfer.sha256) == (len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())
    assert transfer.declared_bytes == len(PAYLOAD)
    assert transfer.verified_length is True
    assert target.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("*.part"))


def test_stream_to_finishes_a_cut_transfer_by_resuming(tmp_path: Path, stub: str) -> None:
    """The MPI endpoint drops long connections; the resume loop must still complete."""

    transfer = Session().stream_to(f"{stub}/capped", tmp_path / "big.zip")
    assert transfer.bytes == len(PAYLOAD)
    assert transfer.segments >= 2
    assert (tmp_path / "big.zip").read_bytes() == PAYLOAD


def test_stream_to_refuses_a_server_that_never_delivers_the_rest(tmp_path: Path, stub: str) -> None:
    """Ignoring Range while declaring a longer file must fail, not yield a short file."""

    with pytest.raises(FetchError, match="still short of"):
        Session().stream_to(f"{stub}/liar", tmp_path / "big.zip", max_segments=4)
    assert not (tmp_path / "big.zip").exists()


def test_stream_to_refuses_html_login_page_served_as_the_asset(
    tmp_path: Path, stub: str
) -> None:
    """A 200 response is not evidence of an asset: the body decides."""
    with pytest.raises(FetchError, match="HTML page"):
        Session().stream_to(f"{stub}/refused", tmp_path / "asset.zip")
    assert not (tmp_path / "asset.zip").exists()
    assert not list(tmp_path.glob("*.part"))


def test_stream_to_refuses_an_empty_body(tmp_path: Path, stub: str) -> None:
    with pytest.raises(FetchError, match="empty body"):
        Session().stream_to(f"{stub}/empty", tmp_path / "asset.zip")
    assert not (tmp_path / "asset.zip").exists()


def test_download_asset_refetches_a_truncated_file_it_already_has(
    tmp_path: Path, stub: str
) -> None:
    """'A non-zero file exists' is not evidence; the container has to open."""

    stale = tmp_path / "asset.zip"
    stale.write_bytes(b"PK\x03\x04\x14\x00" + b"\x00" * 4096)  # header, no central directory
    record = download_asset(
        Session(), RemoteAsset(label="v1.1.0", url=f"{stub}/asset.zip"), tmp_path
    )
    assert record["status"] == "downloaded"
    assert stale.read_bytes() == ZIP_BYTES
    assert record["container"] == ".zip"


def test_download_asset_keeps_a_file_that_already_verifies(tmp_path: Path, stub: str) -> None:
    keep = tmp_path / "asset.zip"
    _make_zip(keep, {"model.pkl": PAYLOAD})
    before = keep.stat().st_mtime_ns
    record = download_asset(
        Session(), RemoteAsset(label="v1.1.0", url=f"{stub}/asset.zip"), tmp_path
    )
    assert "already present" in str(record["status"])
    assert record["container"] == ".zip"
    assert keep.stat().st_mtime_ns == before


def test_discover_reports_unparseable_page_instead_of_guessing(tmp_path: Path, stub: str) -> None:
    from sim2sense_fall.humans.fetch import SiteSpec

    _Stub.bodies = {"/download.php": b"<html><body>nothing here</body></html>"}
    _Stub.types = {"/download.php": "text/html"}
    site = SiteSpec(key="stub", registration_url=f"{stub}/download.php")
    session = Session()
    with pytest.raises(FetchError, match="download links and no login form"):
        discover(
            session,
            site,
            credentials_for_site(site, {"STUB_EMAIL": "a@b.c", "STUB_PASSWORD": PASSWORD}),
        )


def test_discover_distinguishes_a_refused_login_from_an_unauthorised_account(
    tmp_path: Path, stub: str
) -> None:
    """A POST that issues no session cookie is a wrong password, not a missing licence."""

    from sim2sense_fall.humans.fetch import SiteSpec

    _Stub.bodies = {
        "/download.php": b'<form action="/login"><input name="email">'
        b'<input type="password" name="password"></form>',
        "/login": b"<html><body>Wrong password</body></html>",
    }
    _Stub.types = {"/download.php": "text/html", "/login": "text/html"}
    site = SiteSpec(key="stub", registration_url=f"{stub}/download.php")
    with pytest.raises(FetchError, match="no session cookie") as excinfo:
        discover(
            Session(),
            site,
            credentials_for_site(site, {"STUB_EMAIL": "a@b.c", "STUB_PASSWORD": PASSWORD}),
        )
    assert "separate registrations" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Unpacking and provenance
# ---------------------------------------------------------------------------


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return path


def test_verify_container_rejects_a_truncated_zip(tmp_path: Path) -> None:
    cut = tmp_path / "SMPL_python_v.1.1.0.zip"
    cut.write_bytes(b"PK\x03\x04\x14\x00" + b"\x00" * 4096)
    with pytest.raises(FetchError, match="no zip central directory"):
        verify_container(cut)


def test_verify_container_rejects_a_file_lying_about_its_magic(tmp_path: Path) -> None:
    fake = tmp_path / "model.pkl"
    fake.write_bytes(b"not a SMPL model\n")
    with pytest.raises(FetchError, match="starts with"):
        verify_container(fake)
    real = tmp_path / "model.pkl"
    real.write_bytes(b"\x80\x02}q\x00")
    assert verify_container(real) == ".pkl"


def test_verify_container_ignores_formats_it_cannot_check(tmp_path: Path) -> None:
    obj = tmp_path / "smpl_uv.obj"
    obj.write_bytes(b"v 0 0 0\n")
    assert verify_container(obj) == "unchecked"


def test_unpack_archive_extracts_and_hashes_members(tmp_path: Path) -> None:
    archive = _make_zip(
        tmp_path / "SMPL_python.zip",
        {"basicModel_neutral_lbs_10_207_0_v1.1.0.pkl": PAYLOAD, "README.txt": b"hi"},
    )
    records = unpack_archive(archive, tmp_path / "out")
    assert {record["member"] for record in records} == {
        "basicModel_neutral_lbs_10_207_0_v1.1.0.pkl",
        "README.txt",
    }
    landed = tmp_path / "out" / "basicModel_neutral_lbs_10_207_0_v1.1.0.pkl"
    assert landed.read_bytes() == PAYLOAD


@pytest.mark.parametrize("evil", ["../escaped.txt", "sub/../../escaped.txt"])
def test_unpack_archive_refuses_path_traversal(tmp_path: Path, evil: str) -> None:
    archive = _make_zip(tmp_path / "evil.zip", {evil: b"nope"})
    with pytest.raises(FetchError, match="unsafe member path|does not resolve"):
        unpack_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_unpack_archive_returns_nothing_for_a_non_zip(tmp_path: Path) -> None:
    plain = tmp_path / "model.pkl"
    plain.write_bytes(PAYLOAD)
    assert unpack_archive(plain, tmp_path / "out") == []


def test_unpack_archive_refuses_a_broken_archive_instead_of_reporting_nothing(
    tmp_path: Path,
) -> None:
    """A zip that will not open used to yield an empty list, i.e. a silent non-import."""

    cut = tmp_path / "SMPL_python.zip"
    cut.write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(FetchError, match="no zip central directory"):
        unpack_archive(cut, tmp_path / "out")


def test_unpack_archive_handles_tar_xz(tmp_path: Path) -> None:
    import tarfile

    archive = tmp_path / "dmpls.tar.xz"
    with tarfile.open(archive, mode="w:xz") as bundle:
        info = tarfile.TarInfo("dmpl_matrices.pkl")
        info.size = len(PAYLOAD)
        bundle.addfile(info, BytesIO(PAYLOAD))
    records = unpack_archive(archive, tmp_path / "out")
    assert [record["member"] for record in records] == ["dmpl_matrices.pkl"]
    assert (tmp_path / "out" / "dmpl_matrices.pkl").read_bytes() == PAYLOAD


def test_unpack_archive_returns_nothing_for_a_plain_model_file(tmp_path: Path) -> None:
    plain = tmp_path / "model.pkl"
    plain.write_bytes(b"\x80\x02}")
    assert unpack_archive(plain, tmp_path / "out") == []


def test_provenance_record_carries_sha256_and_never_the_password() -> None:
    site = load_sites(REGISTRY)["smpl"]
    credentials = credentials_for_site(
        site, {"SMPL_EMAIL": "me@institution.edu", "SMPL_PASSWORD": PASSWORD}
    )
    asset = RemoteAsset(label="v1.1.0", url="https://download.is.tue.mpg.de/x.zip")
    record = provenance_record(
        asset,
        {"file": "/tmp/x.zip", "bytes": len(PAYLOAD), "sha256": "abc", "status": "downloaded"},
        site=site,
        credentials=credentials,
        unpacked=[{"member": "m.pkl", "bytes": 3, "sha256": "def", "file": "/tmp/out/m.pkl"}],
    )
    assert record["sha256"] == "abc"
    assert record["account"] == "me@institution.edu"
    assert record["unpacked"][0]["member"] == "m.pkl"
    assert PASSWORD not in json.dumps(record)


def test_validate_destination_refuses_a_git_trackable_path(tmp_path: Path) -> None:
    tracked = REPO_ROOT / "configs" / "humans"
    assert validate_destination(REPO_ROOT / "data" / "humans", repo_root=REPO_ROOT) == "git-ignored"
    with pytest.raises(FetchError, match="not git-ignored"):
        validate_destination(tracked, repo_root=REPO_ROOT)
