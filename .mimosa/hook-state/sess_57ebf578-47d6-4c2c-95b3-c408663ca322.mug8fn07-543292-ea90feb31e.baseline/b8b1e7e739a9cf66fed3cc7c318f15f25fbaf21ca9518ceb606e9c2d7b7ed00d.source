#!/usr/bin/env python3
"""Fetch the registration-gated SMPL / AMASS assets with your own MPI account.

Nothing is registered or licensed by this script: it spends credentials you already
hold, reads the download links off the official page, and writes what you are entitled
to into the git-ignored asset root. See ``sim2sense_fall.humans.fetch`` for the
discovery, streaming and provenance rules.

    # 0. what the registry declares as fetchable
    python3 scripts/humans/fetch_assets.py --list-sites

    # 1. validate the credential file without touching the network
    python3 scripts/humans/fetch_assets.py --site smpl --dry-run

    # 2. print exactly what the account can download
    python3 scripts/humans/fetch_assets.py --site smpl --list

    # 3. the SMPL v1.1.0 release plus UV map, unpacked into the asset root
    python3 scripts/humans/fetch_assets.py --site smpl --select 1.1.0 UV --unpack

    # 4. a small AMASS selection
    python3 scripts/humans/fetch_assets.py --site amass --list
    python3 scripts/humans/fetch_assets.py --site amass --select CMU Transition --unpack

Credentials default to ``~/.config/sim2sense-fall/mpg_credentials`` (mode 600, or set
``SIM2SENSE_MPG_CREDENTIALS``). Passwords are never printed, logged, or written to the
provenance log.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
for _directory in (SRC_DIR, REPO_ROOT / "scripts" / "humans"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from sim2sense_fall.humans.fetch import (  # noqa: E402
    CredentialError,
    FetchError,
    Session,
    asset_roots,
    credentials_for_site,
    discover,
    download_asset,
    load_sites,
    provenance_record,
    read_credentials_file,
    select_assets,
    unpack_archive,
    validate_destination,
)

DEFAULT_ASSETS = REPO_ROOT / "configs" / "humans" / "assets.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "humans"
DEFAULT_CREDENTIALS = Path("~/.config/sim2sense-fall/mpg_credentials").expanduser()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch licensed SMPL / AMASS research assets.")
    parser.add_argument(
        "--assets", type=Path, default=DEFAULT_ASSETS, help="asset registry to read"
    )
    parser.add_argument(
        "--site",
        default="smpl",
        help="declared site to use, or 'all' (see --list-sites)",
    )
    parser.add_argument(
        "--list-sites", action="store_true", help="print the declared sites and exit"
    )
    parser.add_argument(
        "--list", action="store_true", help="sign in, print every offered download, fetch nothing"
    )
    parser.add_argument(
        "--select",
        nargs="*",
        default=[],
        help="substrings matched against the offered links, e.g. --select 1.1.0 UV",
    )
    parser.add_argument("--all", action="store_true", help="fetch every offered download")
    parser.add_argument(
        "--page-url", default=None, help="override the page that carries the download links"
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path(os.environ.get("SIM2SENSE_MPG_CREDENTIALS", str(DEFAULT_CREDENTIALS))),
        help=f"KEY=VALUE credential file (default {DEFAULT_CREDENTIALS})",
    )
    parser.add_argument(
        "--ask", action="store_true", help="prompt for any account/password the file leaves empty"
    )
    parser.add_argument(
        "--basic-auth",
        action="store_true",
        help="also send HTTP Basic to the registration host (some MPI endpoints need it)",
    )
    parser.add_argument(
        "--unpack", action="store_true", help="extract downloaded zips into --dest after fetching"
    )
    parser.add_argument("--dest", type=Path, default=None, help="unpack root (default asset root)")
    parser.add_argument(
        "--out", type=Path, default=None, help="download directory (default <root>/_incoming)"
    )
    parser.add_argument(
        "--log", type=Path, default=DEFAULT_OUTPUT_DIR / "asset_fetch_log.json",
        help="provenance log to append to",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the credential file and print the plan, without any network access",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def report_sites(sites: dict[str, Any]) -> None:
    for key, site in sorted(sites.items()):
        print(f"{key}: {site.registration_url}")
        print(f"    {len(site.asset_ids)} declared asset(s): {', '.join(site.asset_ids)}")
        for license_url in site.license_urls:
            print(f"    licence: {license_url}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    registry = args.assets.resolve()
    try:
        sites = load_sites(registry)
    except FetchError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if args.list_sites:
        report_sites(sites)
        return 0

    if args.site == "all":
        chosen = sorted(sites)
    elif args.site in sites:
        chosen = [args.site]
    else:
        print(f"[FAIL] unknown --site {args.site!r}; declared: {', '.join(sorted(sites))}")
        return 1

    try:
        roots = asset_roots(registry, project_root=REPO_ROOT)
    except FetchError as exc:
        print(f"[FAIL] {exc}")
        return 1
    root = roots[0]
    dest = (args.dest or root).expanduser()
    out_dir = (args.out or root / "_incoming").expanduser()

    try:
        values = read_credentials_file(args.credentials.expanduser().resolve())
    except CredentialError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if args.dry_run:
        print(f"[DRY-RUN] registry {registry}, no network access")
        print(f"[DRY-RUN] credential file {args.credentials} parsed {len(values)} key(s)")
        for key in chosen:
            site = sites[key]
            try:
                credentials_for_site(site, values)
                state = "credentials present"
            except CredentialError as exc:
                state = f"MISSING -- {str(exc).splitlines()[0]}"
            print(f"[DRY-RUN] site {key}: {site.registration_url} -> {state}")
        print(f"[DRY-RUN] would fetch into {out_dir}, unpack into {dest}")
        print(f"[DRY-RUN] filters: {args.select or ('--all' if args.all else 'none given')}")
        print("[DRY-RUN] nothing was requested from the server")
        return 0

    failures = 0
    records: list[dict[str, Any]] = []
    for key in chosen:
        site = sites[key]
        prompt = getpass.getpass if args.ask else None
        try:
            credentials = credentials_for_site(site, values, prompt=prompt)
        except CredentialError as exc:
            print(f"[FAIL] {key}: {exc}")
            failures += 1
            continue
        session = Session(
            basic_auth=credentials.basic_header() if args.basic_auth else None,
            basic_auth_hosts=(site.host,) if args.basic_auth else (),
        )
        try:
            page = args.page_url or site.registration_url
            assets = discover(session, site, credentials, page_url=page)
        except FetchError as exc:
            print(f"[FAIL] {key}: {exc}")
            failures += 1
            continue
        picked = select_assets(assets, tuple(args.select), take_all=args.all)
        if args.list:
            print(f"[INFO] {key}: the account is offered {len(assets)} download(s)")
            for asset in assets:
                mark = "*" if asset in picked else " "
                print(f"    [{mark}] {asset.filename:<50} {asset.method} {asset.label[:40]}")
            marked = args.all or bool(args.select)
            print(
                f"[INFO] rows marked * match --select {args.select or '--all'}"
                if marked
                else "[INFO] no --select given: nothing would be fetched"
            )
            continue
        if not args.all and not args.select:
            print(
                f"[FAIL] {key}: refusing to fetch without --select or --all. The AMASS page "
                f"offers {len(assets)} archive(s); downloading them all is tens of gigabytes. "
                "Run --list first and name what you need."
            )
            failures += 1
            continue
        if not picked:
            print(
                f"[FAIL] {key}: --select {args.select} matched none of the {len(assets)} offered "
                "download(s); run with --list to see the real names"
            )
            failures += 1
            continue
        try:
            origin = validate_destination(out_dir, repo_root=REPO_ROOT)
            unpack_root_state = validate_destination(dest, repo_root=REPO_ROOT)
        except FetchError as exc:
            print(f"[FAIL] {key}: {exc}")
            failures += 1
            continue
        print(f"[INFO] {key}: {len(picked)} of {len(assets)} download(s) selected; "
              f"destination {out_dir} ({origin})")
        for asset in picked:
            print(f"[INFO] {key}: {asset.method} {asset.filename}")
            try:
                result = download_asset(session, asset, out_dir, referer=page)
            except FetchError as exc:
                print(f"[FAIL] {exc}")
                failures += 1
                continue
            print(
                f"       {result['bytes']} bytes, sha256 {str(result['sha256'])[:16]}... "
                f"({result['status']}, {result.get('segments', 1)} segment(s), "
                f"{result.get('declared_bytes')} declared, {result.get('container')} container)"
            )
            unpacked: list[dict[str, Any]] = []
            archive = Path(str(result["file"]))
            if args.unpack:
                try:
                    unpacked = unpack_archive(archive, dest)
                except FetchError as exc:
                    print(f"[FAIL] {exc}")
                    failures += 1
                for member in unpacked:
                    print(f"       -> {member['member']} ({member['bytes']} bytes) "
                          f"into {unpack_root_state} root {dest}")
            records.append(
                provenance_record(
                    asset, result, site=site, credentials=credentials, unpacked=unpacked
                )
            )

    if records:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        previous: list[dict[str, Any]] = []
        if args.log.is_file():
            try:
                loaded = json.loads(args.log.read_text(encoding="utf-8"))
                previous = list(loaded.get("fetches") or [])
            except (json.JSONDecodeError, AttributeError, TypeError):
                print(f"[WARN] {args.log} was unreadable; starting a fresh log")
        payload = {
            "version": 1,
            "note": (
                "Provenance for registration-gated research assets: the account that authorised "
                "the download, the source URL, and the SHA-256 of what landed on disk. "
                "Credentials are never recorded."
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fetches": (previous + records)[-400:],
        }
        args.log.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"[INFO] provenance: {args.log} ({len(payload['fetches'])} record(s))")

    if failures:
        print(f"[RESULT] FAILED ({failures} step(s) did not complete)")
        return 1
    print("[RESULT] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
