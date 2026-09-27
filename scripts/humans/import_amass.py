#!/usr/bin/env python3
"""Import local AMASS sequences, retarget them to SMPL, and screen falls.

Example::

    python3 scripts/humans/import_amass.py \
        --root /data/AMASS --out artifacts/humans/amass_import

The command is CPU-only.  It writes retargeted arrays and a manifest with source
hashes, provenance, and the fall-candidate decision.  It never downloads AMASS;
the dataset requires registration and a user supplied local root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_ASSETS,
    DEFAULT_CONFIG,
    DEFAULT_SCENE_CONFIG,
    Checks,
    resolve_spawn_point,
    write_json,
)

from sim2sense_fall.humans.amass import (  # noqa: E402
    annotate_clip,
    load_amass_library,
    screen_amass_clip,
)
from sim2sense_fall.humans.assets import (  # noqa: E402
    load_asset_registry,
    select_body,
)
from sim2sense_fall.humans.config import load_human_config  # noqa: E402
from sim2sense_fall.humans.motion import AMASS_BODY_FRAME  # noqa: E402
from sim2sense_fall.humans.rig import fit_rest_skeleton, plan_human_rig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="local AMASS directory")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/humans/amass_import")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--fall-only", action="store_true", help="write only screened fall candidates"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = Checks()
    try:
        config = load_human_config(args.config)
        registry = load_asset_registry(args.assets, project_root=REPO_ROOT)
        body = select_body(
            registry,
            model_id=config.skeleton.model_asset,
            allow_procedural=bool(config.skeleton.allow_procedural_skeleton),
        )
        if body.has_skin_mesh:
            # AMASS pose parameters live in the body model's own canonical frame, so a
            # frame measured from the template that disagrees with the one used to read
            # the sequences means every joint rotation is being interpreted in the wrong
            # basis -- which still produces a human-sized body, and only shows up as
            # motion that cannot be replayed.
            derived = str(body.model.source_frame)
            if derived != AMASS_BODY_FRAME.describe():
                raise ValueError(
                    f"AMASS sequences are retargeted assuming {AMASS_BODY_FRAME.describe()}, "
                    f"but the loaded body model {config.skeleton.model_asset} was measured in "
                    f"{derived}. Reconcile the two before importing; screening either way "
                    "would report a candidate count for a body nobody simulated."
                )
        rest = (
            fit_rest_skeleton(config, body.model.mesh().rest_skeleton())
            if body.has_skin_mesh
            else None
        )
        spawn = resolve_spawn_point(
            args.scene_config, None, None,
            standing_height_m=config.skeleton.height_m,
        )
        plan = plan_human_rig(config, rest=rest, spawn_xy=spawn)
        loaded = load_amass_library(args.root, limit=args.limit)
        clips = loaded.clips
    except (OSError, ValueError) as exc:
        checks.check("AMASS import inputs", False, f"{type(exc).__name__}: {exc}")
        return checks.report(banner="AMASS import")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for clip in clips.values():
        result = screen_amass_clip(clip, plan)
        tagged = annotate_clip(clip, result)
        item = {
            "clip": tagged.as_dict(),
            "source": tagged.provenance.as_dict(),
            "screen": result.as_dict(),
        }
        if result.accepted or not args.fall_only:
            target = args.out / f"{tagged.clip_id}.npz"
            np.savez_compressed(
                target,
                fps=np.asarray(tagged.fps),
                root_translation=tagged.root_translation,
                root_rotation=tagged.root_rotation,
                joint_rotations=tagged.joint_rotations,
                betas=np.asarray(tagged.betas if tagged.betas is not None else np.zeros(0)),
                allow_pickle=False,
            )
            item["retargeted_npz"] = str(target)
            checks.info(f"{tagged.clip_id}: {'fall candidate' if result.accepted else 'non-fall'}")
        else:
            item["retargeted_npz"] = None
            checks.info(f"{tagged.clip_id}: skipped by --fall-only")
        manifest.append(item)
    output = write_json(
        args.out / "manifest.json",
        {
            "root": str(args.root.resolve()),
            "source_body_frame": AMASS_BODY_FRAME.describe(),
            "files_read": loaded.scanned,
            "unreadable": [{"path": path, "error": error} for path, error in loaded.failures],
            "clips": manifest,
        },
    )
    accepted = sum(bool(item["screen"]["accepted"]) for item in manifest)
    candidates = sum(bool(item["screen"]["fall_candidate"]) for item in manifest)
    checks.check(
        "AMASS sequences imported",
        bool(manifest),
        f"{loaded.scanned} files read ({len(loaded.failures)} unreadable), {len(manifest)} "
        f"screened, {candidates} fall candidates, {accepted} of those expressible by the "
        "shipped single-axis rig",
    )
    checks.info(f"wrote {output}")
    return checks.report(banner="AMASS import")


if __name__ == "__main__":
    raise SystemExit(main())
