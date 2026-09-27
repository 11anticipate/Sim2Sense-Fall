"""Every relative link in the engineering docs must point at a file that exists.

The 2026-09-27 audit found twelve audit documents deleted from the working tree while
`task_plan.md`, `docs/progress.md` and `notes.md` still cited them as the counter-examples
that justify current decisions. A link check is the cheapest guard against that drift,
and it is the check the repo already claims to run by hand ("144 个本地链接").
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# docs/history/** keeps the relative links it was written with; it is an archive of
# superseded audits, not a live entry point, so its internal paths are not checked.
DOCUMENTS = sorted(
    [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md", REPO_ROOT / "notes.md",
     REPO_ROOT / "task_plan.md", *REPO_ROOT.glob("docs/*.md")]
    + [path for path in REPO_ROOT.glob("docs/**/*.md") if "history" not in path.parts]
)
LINK = re.compile(r"\[[^\]]*\]\(([^)#]+)(?:#[^)]*)?\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def _relative_targets(path: Path) -> list[str]:
    return [
        match.group(1).strip()
        for match in LINK.finditer(path.read_text(encoding="utf-8"))
        if not match.group(1).strip().lower().startswith(SKIP_PREFIXES)
    ]


def test_documents_were_found():
    assert len(DOCUMENTS) >= 20, DOCUMENTS


def test_relative_markdown_links_resolve():
    broken: list[str] = []
    for document in DOCUMENTS:
        for target in _relative_targets(document):
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, "broken documentation links:\n  " + "\n  ".join(sorted(broken))
