#!/usr/bin/env python3
"""Render the top symmetric-squat candidates from the source screen.

Consumes ``artifacts/humans/squat_source_screen.json`` (from
``screen_squat_sources.py``), retargets and skins each shortlisted frame,
and writes one PNG per candidate for visual selection. CPU-only preview:
the frames are reference poses, not physics results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))

from common import DEFAULT_MOTIONS, REPO_ROOT, load_inputs  # noqa: E402

from sim2sense_fall.humans.amass import load_amass_clip_by_id, normalize_root_motion  # noqa: E402
from sim2sense_fall.humans.assets import select_body  # noqa: E402
from sim2sense_fall.humans.mesh_sequence import fit_mesh_to_rest_joints  # noqa: E402
from sim2sense_fall.humans.rig import (  # noqa: E402
    fit_rest_skeleton,
    forward_kinematics,
    joint_values_from_clip,
    plan_human_rig,
)
from sim2sense_fall.humans.skinning import skin_with_link_poses  # noqa: E402
from sim2sense_fall.humans.teleop import load_keyboard_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screen", type=Path,
                        default=REPO_ROOT / "artifacts/humans/squat_source_screen.json")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "artifacts/humans/squat_candidates")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_keyboard_config(REPO_ROOT / "configs/humans/keyboard.yaml", REPO_ROOT)
    config, registry, _ = load_inputs(config_path=settings["rig"],
                                      assets_path=settings["assets"],
                                      motions_path=DEFAULT_MOTIONS)
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    plan = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    mesh = fit_mesh_to_rest_joints(mesh, np.asarray(plan.rest_joint_positions))
    dof_names = list(plan.dof_names)

    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    rows = screen["top"][: args.count]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, object] = {}
    for order, row in enumerate(rows):
        clip_id, frame = row["clip"], row["frame"]
        if clip_id not in cache:
            cache[clip_id] = normalize_root_motion(
                load_amass_clip_by_id(REPO_ROOT / "data/humans", clip_id)
            )
        clip = cache[clip_id]
        values, _ = joint_values_from_clip(clip, frame, plan)
        poses = forward_kinematics(plan, dict(zip(dof_names, values, strict=False)),
                                   root_position=(0.0, 0.0, 0.0),
                                   root_rotation=np.zeros(3))
        links = {name: (t.rotation, t.translation) for name, t in poses.items()}
        verts = skin_with_link_poses(mesh, links)
        -min(verts[:, 2].min(), 0.0)
        fig = plt.figure(figsize=(4.6, 4.6))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                          triangles=mesh.faces, color="0.78",
                          edgecolor="none", shade=True)
        axis.set_box_aspect((np.ptp(verts[:, 0]), np.ptp(verts[:, 1]),
                             np.ptp(verts[:, 2])))
        axis.view_init(elev=16, azim=-65)
        axis.set_axis_off()
        axis.set_title(f"#{order} {clip_id} f{frame}\n"
                       f"pelvis {row['pelvis_ratio']:.0%} knee {row['knee_deg']:.0f} deg",
                       fontsize=9)
        out = args.out_dir / f"cand_{order:02d}_{clip_id}_f{frame}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"{out.name}  (score {row['score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
