"""Authenticated fetch for the registration-gated SMPL / AMASS research assets.

Both families are non-commercial research releases behind a Max Planck account: a
person has to register and accept the licence before any file exists for them. This
module therefore never invents a download URL. It starts from the ``registration_url``
already declared in :file:`configs/humans/assets.yaml`, signs in with credentials read
from a mode-600 file, enumerates the links the page actually offers, and streams the
selected ones to disk while hashing them.

Design constraints worth knowing before editing:

* **Nothing is registered or accepted here.** The licence decision belongs to a human;
  this only spends credentials that already carry it.
* **Discovery over hardcoding.** The download page is parsed, so a renamed archive or a
  changed ``sfile`` shows up as a different link list rather than as a 404 against a
  path this repository guessed.
* **Credentials never leave memory.** They are not printed, logged, or written into the
  provenance record; only the account e-mail is recorded, which is what makes a
  re-download attributable to the licence that authorised it.
* **Licensed bytes stay untracked.** :func:`assert_path_not_tracked` refuses to write
  anywhere Git could pick up, so ``data/humans`` must be ignored.
"""

from __future__ import annotations

import base64
import hashlib
import http.cookiejar
import logging
import os
import re
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

__all__ = [
    "ASSET_SUFFIXES",
    "CredentialError",
    "Credentials",
    "FetchError",
    "RemoteAsset",
    "Session",
    "SiteSpec",
    "Transfer",
    "asset_roots",
    "credentials_for_site",
    "discover",
    "download_asset",
    "load_sites",
    "parse_page",
    "provenance_record",
    "read_credentials_file",
    "select_assets",
    "unpack_archive",
    "validate_destination",
    "verify_container",
]

logger = logging.getLogger(__name__)

#: Archive/object extensions treated as data rather than page chrome.
ASSET_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".npz", ".pkl", ".obj", ".gz", ".tar")

#: Links containing these are navigation, licence text, or account plumbing.
NAVIGATION_HINTS = ("license", "licence", "terms", "account", "logout", "login", "citation", "help")

USER_AGENT = "Sim2Sense-Fall asset fetch/1.0 (licensed research use)"

#: A body that starts like this is an HTML page, not the requested archive.
_HTML_SNIFF = re.compile(rb"<\s*(?:!doctype html|html|head|body|form)\b", re.IGNORECASE)


class FetchError(RuntimeError):
    """A fetch step failed for a reason the operator can act on."""


class CredentialError(FetchError):
    """The credential file is absent, unreadable, or missing this site's pair."""


# ---------------------------------------------------------------------------
# Registry-derived site descriptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteSpec:
    """One release site as declared by the asset registry."""

    key: str
    registration_url: str
    license_urls: tuple[str, ...] = ()
    asset_ids: tuple[str, ...] = ()

    @property
    def origin(self) -> str:
        """Scheme + host of the registration page, i.e. where the session cookie lives."""
        parsed = urllib.parse.urlsplit(self.registration_url)
        if not parsed.scheme or not parsed.netloc:
            raise FetchError(f"registration URL for {self.key!r} is not absolute: {parsed!r}")
        return f"{parsed.scheme}://{parsed.netloc}"

    @property
    def host(self) -> str:
        return str(urllib.parse.urlsplit(self.origin).netloc)

    @property
    def email_key(self) -> str:
        return f"{self.key.upper()}_EMAIL"

    @property
    def password_key(self) -> str:
        return f"{self.key.upper()}_PASSWORD"


def _registry_document(assets_path: Path) -> dict[str, Any]:
    import yaml

    if not assets_path.is_file():
        raise FetchError(f"no asset registry at {assets_path}")
    document = yaml.safe_load(assets_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise FetchError(f"{assets_path} did not parse to a mapping")
    return document


def load_sites(assets_path: Path) -> dict[str, SiteSpec]:
    """Group every declared asset by the site its ``registration_url`` points at."""

    document = _registry_document(assets_path)
    grouped: dict[str, dict[str, Any]] = {}
    for section in ("models", "auxiliary", "motions"):
        for entry in document.get(section) or []:
            if not isinstance(entry, Mapping):
                continue
            url = str(entry.get("registration_url") or "").strip()
            if not url:
                continue
            bucket = grouped.setdefault(url, {"licenses": set(), "ids": []})
            license_url = str(entry.get("license_url") or "").strip()
            if license_url:
                bucket["licenses"].add(license_url)
            bucket["ids"].append(str(entry.get("id") or entry.get("dataset") or "unknown"))
    if not grouped:
        raise FetchError(
            f"{assets_path} declares no registration_url, so there is nothing to fetch"
        )
    sites: dict[str, SiteSpec] = {}
    for url, bucket in grouped.items():
        host = urllib.parse.urlsplit(url).netloc.lower()
        label = host.split(".")[0]
        sites[label] = SiteSpec(
            key=label,
            registration_url=url,
            license_urls=tuple(sorted(bucket["licenses"])),
            asset_ids=tuple(sorted(bucket["ids"])),
        )
    return sites


def asset_roots(assets_path: Path, *, project_root: Path) -> list[Path]:
    """Declared search roots, with ``$PROJECT`` expanded."""

    raw = _registry_document(assets_path).get("roots") or []
    if not raw:
        raise FetchError(f"{assets_path} declares no 'roots'")
    return [
        Path(str(entry).replace("$PROJECT", str(project_root))).expanduser() for entry in raw
    ]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Credentials:
    """One account that has already accepted the site's licence."""

    email: str
    password: str = field(repr=False, default="")

    def __str__(self) -> str:  # pragma: no cover - guards accidental logging
        return f"Credentials(email={self.email!r}, password=<redacted>)"

    def basic_header(self) -> str:
        token = base64.b64encode(f"{self.email}:{self.password}".encode()).decode("ascii")
        return f"Basic {token}"


def read_credentials_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines, refusing a file the rest of the machine can read."""

    if not path.is_file():
        raise CredentialError(
            f"no credential file at {path}\n"
            "  create it with one KEY=VALUE pair per site, e.g.\n"
            "    SMPL_EMAIL=you@institution.edu\n"
            "    SMPL_PASSWORD=...\n"
            f"  then: chmod 600 {path}"
        )
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise CredentialError(
            f"{path} grants group/other access (mode {mode:04o}); run chmod 600 {path} first"
        )
    values: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        key, sep, value = text.partition("=")
        if not sep:
            raise CredentialError(f"{path}:{lineno}: expected KEY=VALUE, got {text[:40]!r}")
        values[key.strip().upper()] = value.strip()
    return values


Prompt = Callable[[str], str] | None


def credentials_for_site(
    site: SiteSpec, values: Mapping[str, str], *, prompt: Prompt = None
) -> Credentials:
    """Extract this site's pair, optionally asking for whatever the file left empty."""

    email = str(values.get(site.email_key) or "")
    password = str(values.get(site.password_key) or "")
    if prompt is not None:
        if not email:
            email = prompt(f"{site.key} account e-mail for {site.host}: ").strip()
        if not password:
            password = prompt(f"{site.key} password for {email or site.host}: ")
    if not email or not password:
        license_hint = site.license_urls[0] if site.license_urls else site.registration_url
        raise CredentialError(
            f"{site.email_key} / {site.password_key} is empty or missing. Register at "
            f"{site.origin}/account/ and accept the licence at {license_hint}, then fill the "
            "credential file (or prompt for it)."
        )
    return Credentials(email=email, password=password)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Response:
    """A completed GET/POST: where it landed and what came back."""

    url: str
    status: int
    content_type: str
    body: bytes

    @property
    def looks_like_html(self) -> bool:
        return _HTML_SNIFF.search(self.body[:4096]) is not None


@dataclass(frozen=True)
class Transfer:
    """One completed download: how much arrived, and whether the server agreed."""

    bytes: int
    sha256: str
    content_type: str
    declared_bytes: int | None = None
    segments: int = 1

    @property
    def verified_length(self) -> bool:
        return self.declared_bytes is None or self.bytes >= int(self.declared_bytes)


class Session:
    """Cookie-aware HTTP client. One instance per site keeps sessions isolated."""

    def __init__(
        self,
        *,
        timeout: float = 90.0,
        basic_auth: str | None = None,
        basic_auth_hosts: Sequence[str] = (),
    ) -> None:
        self.cookiejar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookiejar)
        )
        self.timeout = float(timeout)
        self.basic_auth = basic_auth
        self.basic_auth_hosts = {str(host).lower() for host in basic_auth_hosts}
        self._csrf: str | None = None

    # -- header plumbing ------------------------------------------------ --

    def _headers(
        self, url: str, extra: Mapping[str, str] | None, *, posting: bool
    ) -> dict[str, str]:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/octet-stream;q=0.9,*/*;q=0.8",
        }
        host = urllib.parse.urlsplit(url).netloc.lower()
        if self.basic_auth and host in self.basic_auth_hosts:
            headers["Authorization"] = self.basic_auth
        if posting and self._csrf:
            headers["X-CSRFToken"] = self._csrf
        headers.update({str(k): str(v) for k, v in (extra or {}).items() if v})
        return headers

    def _capture_csrf(self) -> None:
        for cookie in self.cookiejar:
            if cookie.name.lower() in {"csrftoken", "csrf_token", "_csrf"}:
                self._csrf = cookie.value

    # -- requests ------------------------------------------------------- --

    def request(
        self,
        url: str,
        *,
        data: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        body = None
        send = dict(headers or {})
        if data is not None:
            body = urllib.parse.urlencode(dict(data)).encode()
            send.setdefault("Referer", url)
        request = urllib.request.Request(
            url, data=body, headers=self._headers(url, send, posting=data is not None)
        )
        logger.debug("%s %s", "POST" if body else "GET", url)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = Response(
                    url=str(response.geturl()),
                    status=int(response.status),
                    content_type=str(response.headers.get_content_type()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            snippet = exc.read()[:300].decode("utf-8", "replace").strip()
            raise FetchError(f"HTTP {exc.code} for {url}: {snippet[:180]}") from exc
        except urllib.error.URLError as exc:
            raise FetchError(f"could not reach {url}: {exc.reason}") from exc
        self._capture_csrf()
        return payload

    def stream_to(
        self,
        url: str,
        target: Path,
        *,
        headers: Mapping[str, str] | None = None,
        referer: str | None = None,
        first_chunk_probe: int = 4096,
        max_segments: int = 200,
    ) -> Transfer:
        """Write ``url`` to ``target``, resuming until the declared length is complete.

        The MPI endpoint closes long connections mid-file, which a plain read-until-EOF
        loop reports as a finished download: the first SMPL v1.1.0 pull landed 36 MB of a
        larger archive with no zip central directory at all. A transfer is therefore
        accepted only when the byte count reaches what the server declared, via
        ``Content-Length`` or the total in a 206 ``Content-Range``, continuing with
        ``Range`` requests as many times as ``max_segments`` allows.

        Streams rather than reading into memory because the larger AMASS subsets are
        multi-gigabyte. The first chunk is sniffed for HTML, which is how a refused
        session shows up when the server answers 200 with a login page.
        """

        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        content_type = ""
        declared: int | None = None
        segments = 0
        stalled: Exception | None = None
        for _ in range(max(1, int(max_segments))):
            have = part.stat().st_size if part.is_file() else 0
            send = dict(headers or {})
            if referer:
                send["Referer"] = referer
            if have:
                send["Range"] = f"bytes={have}-"
            request = urllib.request.Request(
                url, headers=self._headers(url, send, posting=False)
            )
            probe = b""
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    status = int(response.status)
                    ctype = str(response.headers.get_content_type())
                    content_type = content_type or ctype
                    total, ignored_range = _declared_total(
                        response.headers, status=status, already=have
                    )
                    if total is not None:
                        declared = total if declared is None else max(declared, total)
                    if ignored_range:  # the server restarted the body: begin over
                        part.unlink(missing_ok=True)
                        have = 0
                    with part.open("ab" if have else "wb") as handle:
                        seen = have
                        while True:
                            chunk = response.read(1 << 16)
                            if not chunk:
                                break
                            if have == 0 and seen <= first_chunk_probe:
                                probe = (probe + chunk)[:first_chunk_probe]
                                if _HTML_SNIFF.search(probe):
                                    part.unlink(missing_ok=True)
                                    raise FetchError(
                                        f"{url} returned an HTML page ({ctype}), not the asset; "
                                        "the session was most likely refused"
                                    )
                            handle.write(chunk)
                            seen += len(chunk)
                segments += 1
            except FetchError:
                raise
            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
                # A dropped connection is the case the resume loop exists for: keep the
                # partial file and ask for the rest, unless it is a real refusal.
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                    part.unlink(missing_ok=True)
                    snippet = exc.read()[:300].decode("utf-8", "replace").strip()
                    raise FetchError(f"HTTP {exc.code} for {url}: {snippet[:180]}") from exc
                stalled = exc
                logger.warning(
                    "transfer of %s interrupted at %d bytes: %s",
                    url,
                    part.stat().st_size if part.is_file() else 0,
                    exc,
                )
            have = part.stat().st_size if part.is_file() else 0
            if declared is not None and have >= declared:
                break
        else:
            if declared is not None:
                raise FetchError(
                    f"{url}: still short of {declared} declared bytes after {segments} segment(s)"
                )
        have = part.stat().st_size if part.is_file() else 0
        if have == 0:
            part.unlink(missing_ok=True)
            raise FetchError(f"{url} returned an empty body ({stalled})")
        if declared is not None and have < declared:
            raise FetchError(
                f"{url}: got {have} of {declared} declared bytes after {segments} segment(s); "
                f"the partial file is kept at {part} for a re-run"
            )
        digest = _sha256_of(part)
        part.replace(target)
        self._capture_csrf()
        return Transfer(
            bytes=have, sha256=digest, content_type=content_type,
            declared_bytes=declared, segments=segments,
        )


# ---------------------------------------------------------------------------
# Page parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteAsset:
    """One download the page offers."""

    label: str
    url: str
    fields: tuple[tuple[str, str], ...] = ()

    @property
    def method(self) -> str:
        return "POST" if self.fields else "GET"

    @property
    def filename(self) -> str:
        query = urllib.parse.urlsplit(self.url).query
        params = urllib.parse.parse_qs(query)
        for key in ("sfile", "file", "filename"):
            values = params.get(key)
            if values:
                return Path(urllib.parse.unquote(values[0])).name
        path_name = Path(urllib.parse.urlsplit(self.url).path).name
        if path_name.lower().endswith(ASSET_SUFFIXES):
            return path_name
        for _name, value in self.fields:
            if value.lower().endswith(ASSET_SUFFIXES):
                return Path(urllib.parse.unquote(value)).name
        # A script URL with no file parameter: the anchor text is the only name left.
        return re.sub(r"\W+", "_", self.label).strip("_") or "asset.bin"


@dataclass(frozen=True)
class LoginForm:
    """A password form found on the page, ready to be filled."""

    action: str
    fields: Mapping[str, str]
    email_field: str
    password_field: str


@dataclass(frozen=True)
class PageOffer:
    """What one page told us: downloads, and whether it wanted a login."""

    url: str
    assets: tuple[RemoteAsset, ...]
    login_form: LoginForm | None
    asks_password: bool


class _LinkCollector(HTMLParser):
    """Collects anchors and forms while noting whether the page asked for a password."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.forms: list[dict[str, Any]] = []
        self.asks_password = False
        self._form: dict[str, Any] | None = None
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {str(key).lower(): (value or "") for key, value in attrs}
        if tag == "a" and values.get("href"):
            self._href = values["href"]
            self._text = []
        elif tag == "form":
            self._form = {
                "action": values.get("action", ""),
                "method": values.get("method", "get").lower(),
                "inputs": {},
                "password_field": "",
            }
        elif tag == "input" and self._form is not None:
            name = values.get("name", "")
            kind = values.get("type", "text").lower()
            if kind == "password":
                self.asks_password = True
                self._form["password_field"] = name or "password"
            elif name and kind in {"hidden", "text", "email"}:
                self._form["inputs"][name] = values.get("value", "")
        elif tag == "button" and self._form is not None and values.get("name"):
            self._form["inputs"].setdefault(values["name"], values.get("value", ""))

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            label = " ".join(" ".join(self._text).split()) or self._href
            self.links.append((label, self._href))
            self._href = None
        elif tag == "form" and self._form is not None:
            self.forms.append(self._form)
            self._form = None


def _is_asset_link(url: str) -> bool:
    lowered = url.lower()
    if any(hint in lowered for hint in NAVIGATION_HINTS):
        return False
    query = urllib.parse.urlsplit(lowered).query
    return "sfile=" in query or lowered.endswith(ASSET_SUFFIXES)


def parse_page(raw: bytes, base_url: str) -> PageOffer:
    """Extract downloadable links and a login form from one page."""

    collector = _LinkCollector()
    collector.feed(raw.decode("utf-8", "replace"))
    assets: list[RemoteAsset] = []
    seen: set[str] = set()
    for label, href in collector.links:
        if href.startswith(("javascript:", "#", "mailto:")):
            continue
        url = urllib.parse.urljoin(base_url, href)
        if not _is_asset_link(url) or url in seen:
            continue
        seen.add(url)
        assets.append(RemoteAsset(label=label or Path(url).name, url=url))
    login_form: LoginForm | None = None
    for form in collector.forms:
        password_field = str(form.get("password_field") or "")
        if not password_field:
            continue
        fields = dict(form["inputs"])
        email_field = next(
            (name for name in fields if any(t in name.lower() for t in ("email", "user", "login"))),
            "username",
        )
        login_form = LoginForm(
            action=urllib.parse.urljoin(base_url, str(form.get("action") or base_url)),
            fields=fields,
            email_field=email_field,
            password_field=password_field,
        )
        break
    return PageOffer(
        url=base_url,
        assets=tuple(assets),
        login_form=login_form,
        asks_password=collector.asks_password,
    )


def discover(
    session: Session,
    site: SiteSpec,
    credentials: Credentials,
    *,
    page_url: str | None = None,
) -> tuple[RemoteAsset, ...]:
    """Sign in if needed and return every download the account is offered."""

    page = session.request(page_url or site.registration_url)
    offer = parse_page(page.body, page.url)
    if offer.assets and not offer.asks_password and not page.looks_like_html:
        return offer.assets
    if offer.login_form is None:
        raise FetchError(
            f"{site.key}: {page.url} returned HTTP {page.status} ({page.content_type}) with "
            f"{len(offer.assets)} download links and no login form. Either the account is not "
            "authorised on this page, or its layout is not understood: open it in a browser, "
            "read the real link, and pass it with --page-url."
        )
    fields = dict(offer.login_form.fields)
    fields[offer.login_form.email_field] = credentials.email
    fields[offer.login_form.password_field] = credentials.password
    after = session.request(offer.login_form.action, data=fields)
    signed_in = parse_page(after.body, after.url)
    if not signed_in.assets:
        # A successful MPI sign-in redirects to the landing page rather than back to the
        # table of links, so an empty first parse proves nothing yet: ask the page that
        # carries the links for its own account.
        retry = session.request(page_url or site.registration_url)
        signed_in = parse_page(retry.body, retry.url)
    if not signed_in.assets:
        session_cookie = next(
            (
                cookie.name
                for cookie in session.cookiejar
                if any(hint in cookie.name.lower() for hint in ("session", "phpbb", "jsid", "auth"))
            ),
            None,
        )
        if session_cookie is None:
            raise FetchError(
                f"{site.key}: {offer.login_form.action} accepted the POST (HTTP {after.status}) "
                f"but issued no session cookie, and landed on {after.url}. That is a refused "
                "login: wrong account or password, or the licence has not been accepted on this "
                "site yet. Note that smpl.is.tue.mpg.de and amass.is.tue.mpg.de are separate "
                "registrations -- an account on one is not an account on the other."
            )
        raise FetchError(
            f"{site.key}: signed in (session cookie {session_cookie!r}) but {signed_in.url} "
            f"lists no download links. The account may not be authorised for this release yet; "
            "open the page in a browser, read the real link, and pass it with --page-url."
        )
    return signed_in.assets


def select_assets(
    assets: Sequence[RemoteAsset], patterns: Sequence[str], *, take_all: bool = False
) -> tuple[RemoteAsset, ...]:
    """Case-insensitive substring match over label, file name and URL."""

    if take_all or not patterns:
        return tuple(assets)
    lowered = [pattern.lower() for pattern in patterns]
    return tuple(
        asset
        for asset in assets
        if any(
            needle in f"{asset.label} {asset.filename} {asset.url}".lower() for needle in lowered
        )
    )


# ---------------------------------------------------------------------------
# Download, unpack, provenance
# ---------------------------------------------------------------------------


def _safe_name(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", filename).strip("._")
    return name or "asset.bin"


def _sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _compound_suffix(path: Path) -> str:
    """The suffix that matters, keeping ``.tar.xz`` intact."""

    name = path.name.lower()
    for known in (".tar.xz", ".tar.gz", ".tar.bz2"):
        if name.endswith(known):
            return known
    return path.suffix.lower()


def _declared_total(headers: Any, *, status: int, already: int) -> tuple[int | None, bool]:
    """The whole-file size the server declared, and whether our Range request was ignored.

    ``already`` is how many bytes are on disk. A 200 answer to a ranged request means the
    body restarts from zero, so the partial file has to be replaced rather than appended.
    """

    content_range = str(headers.get("Content-Range") or "")
    match = re.search(r"/(\d+)\s*$", content_range)
    length = headers.get("Content-Length")
    total_from_length = int(length) if length is not None and str(length).isdigit() else None
    if status == 206:
        if match:
            return int(match.group(1)), False
        return (already + total_from_length) if total_from_length is not None else None, False
    if status == 200:
        if already and total_from_length is not None:
            return total_from_length, True
        return total_from_length, False
    return None, False


ZIP_LIKE = frozenset({".zip", ".npz"})
TAR_LIKE = frozenset({".tar", ".tar.xz", ".tar.gz", ".tar.bz2"})
_MAGIC = {
    ".xz": (b"\xfd7zXZ\x00",),
    ".gz": (b"\x1f\x8b",),
    ".pdf": (b"%PDF",),
    ".pkl": (b"\x80",),
}


def verify_container(path: Path, *, check_crc: bool = False) -> str:
    """Require a downloaded file to be the container its own name claims.

    This is the check the first SMPL pull needed: a dropped connection left 36 MB of a
    larger archive on disk, which opens as bytes but has no zip central directory, so it
    looked like a finished download. Returns a short label for the provenance record.
    """

    kind = _compound_suffix(path)
    if kind in ZIP_LIKE:
        if not zipfile.is_zipfile(path):
            raise FetchError(
                f"{path.name}: named {kind} but has no zip central directory, so it is "
                f"truncated or not an archive at all ({path.stat().st_size} bytes)"
            )
        if check_crc:
            with zipfile.ZipFile(path) as bundle:
                broken = bundle.testzip()
            if broken is not None:
                raise FetchError(f"{path.name}: member {broken!r} fails its CRC check")
        return kind
    magic = _MAGIC.get(kind)
    if magic is None:
        return "unchecked"
    with path.open("rb") as handle:
        head = handle.read(8)
    if not any(head.startswith(prefix) for prefix in magic):
        raise FetchError(
            f"{path.name}: named {kind} but starts with {head[:5]!r}, not {magic[0][:5]!r}"
        )
    return kind


def download_asset(
    session: Session,
    asset: RemoteAsset,
    dest_dir: Path,
    *,
    referer: str | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> dict[str, Any]:
    """Fetch one asset into ``dest_dir``, insisting on length and container before truth.

    A file already on disk is only kept once it passes the same content checks a fresh
    download gets; an earlier revision accepted any non-zero size, which is how a
    truncated archive survived a re-run.
    """

    target = dest_dir / _safe_name(asset.filename)
    if target.is_file() and target.stat().st_size > 0:
        try:
            container = verify_container(target)
        except FetchError as exc:
            logger.warning("discarding %s: %s", target.name, exc)
            target.unlink(missing_ok=True)
        else:
            return {
                "file": str(target),
                "bytes": target.stat().st_size,
                "sha256": _sha256_of(target),
                "container": container,
                "status": "already present and structurally verified, not re-fetched",
            }
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            transfer = session.stream_to(asset.url, target, referer=referer or asset.url)
            container = verify_container(target)
            return {
                "file": str(target),
                "bytes": transfer.bytes,
                "sha256": transfer.sha256,
                "content_type": transfer.content_type,
                "declared_bytes": transfer.declared_bytes,
                "segments": transfer.segments,
                "container": container,
                "status": "downloaded",
            }
        except (FetchError, OSError) as exc:
            last_error = exc
            logger.warning("attempt %d for %s failed: %s", attempt, asset.filename, exc)
            if attempt < retries:
                time.sleep(retry_delay * attempt)
    raise FetchError(
        f"{asset.filename}: download did not complete after {retries} attempt(s): {last_error}"
    )


def _extract_zip(bundle: zipfile.ZipFile, root: Path, archive: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for info in bundle.infolist():
        if info.is_dir():
            continue
        target = (root / Path(info.filename)).resolve()
        _reject_escape(info.filename, target, root, archive)
        records.append(_land(target, bundle.read(info), info.filename))
    return records


def _extract_tar(bundle: Any, root: Path, archive: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for member in bundle.getmembers():
        if not member.isfile():
            continue
        handle = bundle.extractfile(member)
        if handle is None:
            continue
        target = (root / Path(member.name)).resolve()
        _reject_escape(member.name, target, root, archive)
        records.append(_land(target, handle.read(), member.name))
    return records


def _reject_escape(name: str, target: Path, root: Path, archive: Path) -> None:
    """Refuse an extracted member that would land outside the destination."""

    member = Path(name)
    if member.is_absolute() or ".." in member.parts:
        raise FetchError(f"{archive.name}: refusing unsafe member path {name!r}")
    if target == root or root not in target.parents:
        raise FetchError(f"{archive.name}: member {name!r} does not resolve under {root}")


def _land(target: Path, payload: bytes, member_name: str) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {
        "member": str(member_name),
        "file": str(target),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def unpack_archive(archive: Path, dest: Path) -> list[dict[str, Any]]:
    """Extract a zip or tar archive into ``dest``, refusing members that escape it.

    An archive named as such that will not open is an error, not an empty result: the
    stage-7 SMPL archive arrived truncated, and silently returning no members was what
    let a 36 MB fragment count as a delivered model file.
    """

    kind = _compound_suffix(archive)
    root = dest.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if kind == ".npz":
        # A packed NumPy archive, not a bundle to explode: np.load reads it whole, and
        # extracting it would leave loose .npy files the loader cannot resolve.
        return []
    if kind in ZIP_LIKE:
        if not zipfile.is_zipfile(archive):
            raise FetchError(
                f"{archive.name}: named {kind} but has no zip central directory "
                f"({archive.stat().st_size} bytes); re-fetch it"
            )
        with zipfile.ZipFile(archive) as bundle:
            records = _extract_zip(bundle, root, archive)
    elif kind in TAR_LIKE or kind in {".xz", ".gz"}:
        import tarfile

        try:
            with tarfile.open(archive, mode="r:*") as bundle:
                records = _extract_tar(bundle, root, archive)
        except tarfile.TarError as exc:
            raise FetchError(f"{archive.name}: could not read the tar stream: {exc}") from exc
    else:
        return []
    logger.info("unpacked %d member(s) from %s into %s", len(records), archive.name, root)
    return records


def provenance_record(
    asset: RemoteAsset,
    download: Mapping[str, Any],
    *,
    site: SiteSpec,
    credentials: Credentials,
    unpacked: Sequence[Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the audit trail for one fetch. Never includes a password."""

    timestamp = now or datetime.now(timezone.utc)
    return {
        "site": site.key,
        "account": credentials.email,
        "label": asset.label,
        "requested_url": asset.url,
        "method": asset.method,
        "file": download.get("file"),
        "bytes": download.get("bytes"),
        "declared_bytes": download.get("declared_bytes"),
        "segments": download.get("segments"),
        "container": download.get("container"),
        "sha256": download.get("sha256"),
        "status": download.get("status"),
        "licence_urls": list(site.license_urls),
        "registration_url": site.registration_url,
        "licence_accepted_by": "the account holder, before this fetch",
        "fetched_at": timestamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "unpacked": [dict(entry) for entry in unpacked],
    }


def _ignored_by_git(path: Path, *, repo_root: Path) -> bool | None:
    """True if Git ignores ``path``, False if it does not, None if Git could not say.

    ``data/humans/`` is written with a trailing slash, so Git reports the directory
    itself as not ignored while everything under it is. Both spellings are therefore
    asked about, and only the plain-path answer counts as 'this could be committed'.
    """

    probes = (str(path), f"{path}{os.sep}")
    returncode: list[int] = []
    for probe in probes:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "check-ignore", "-q", "--", probe],
                capture_output=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("git check-ignore unavailable (%s) for %s", exc, probe)
            return None
        if result.returncode == 0:
            return True
        returncode.append(result.returncode)
    if any(code not in (0, 1) for code in returncode):
        return None
    return False


def validate_destination(path: Path, *, repo_root: Path) -> str:
    """Require the write target to be git-ignored, so licensed data cannot be committed.

    Returns a short description of how the check was settled, so the caller can record
    it. Raises when Git says the path is trackable; only reports when Git itself is
    unavailable, because an un-checkable destination is still an un-tracked machine.
    """

    verdict = _ignored_by_git(path, repo_root=repo_root)
    if verdict is None:
        logger.warning("could not ask Git whether %s is ignored; writing anyway", path)
        return "unchecked: git unavailable"
    if not verdict:
        raise FetchError(
            f"{path} is not git-ignored. Licensed SMPL/AMASS data must never be committed; "
            "add an ignore rule or choose a destination under data/humans."
        )
    return "git-ignored"
