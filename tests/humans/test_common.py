"""Provenance path keys must survive symlinked checkout spellings.

The GUI entry point is documented as runnable from any directory, and the
repository can be reached through more than one path (a real checkout plus
symlinked spellings to it). ``Path.relative_to`` compares raw strings, so the
provenance builder comparing an unresolved launch path against the resolved
``REPO_ROOT`` raised ``ValueError`` before the GUI ever opened. The fix
canonicalizes both sides via ``repo_relative``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))

from common import repo_relative  # noqa: E402


def test_repo_relative_accepts_symlinked_launch_path(tmp_path):
    link = tmp_path / "repo-link"
    link.symlink_to(REPO_ROOT, target_is_directory=True)
    assert repo_relative(link / "scripts/humans/keyboard.py") == "scripts/humans/keyboard.py"


def test_repo_relative_is_stable_for_resolved_paths():
    assert repo_relative(REPO_ROOT / "scripts/humans/keyboard.py") == "scripts/humans/keyboard.py"


def test_repo_relative_rejects_paths_outside_the_repo(tmp_path):
    outside = tmp_path / "elsewhere.py"
    outside.write_text("", encoding="utf-8")
    try:
        repo_relative(outside)
    except ValueError:
        pass
    else:
        raise AssertionError("paths outside REPO_ROOT must fail fast, not mislabel provenance")
