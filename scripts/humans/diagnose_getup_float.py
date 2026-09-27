"""Isolate why the G get-up floats: commanded trajectory vs. measured floor contact.

The recovery replays an AMASS lying-to-standing clip whose root height is one
absolute curve, grounded **once** on the clip's first frame
(:func:`sim2sense_fall.humans.actions.load_action_clip`). The posture path
(:meth:`ActionState.apply`) instead re-grounds every commanded configuration by
contact closure. This script measures both sides of that gap on the real assets
and on an already-recorded session, so the fix targets the actual cause:

  A. Reference side (CPU, no simulator): for every clip frame, compare the
     clip's absolute root height with the height that puts the lowest declared
     support capsule exactly on the floor, and report which link that support is.
  B. Measured side: from a keyboard session artifact, attribute every floor
     contact impulse to the body link that made it, and report how much of the
     recovery has no floor contact at all plus the lowest skinned vertex.

CPU only: no Isaac Sim, no Sionna, no simulator state.

Usage:
    python3 scripts/humans/diagnose_getup_float.py \
        --artifact artifacts/diag_getup_slerp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim2sense_fall.humans.actions import load_action_clip
from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.contact_control import capsule_bottom, fit_collision_capsules
from sim2sense_fall.humans.rig import fit_rest_skeleton, forward_kinematics, plan_human_rig
from sim2sense_fall.humans.teleop import load_keyboard_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/humans/keyboard.yaml"
DEFAULT_MOTIONS = REPO_ROOT / "configs/humans/motions.yaml"
# Everything the body may legitimately rest on during a lying-to-standing
# recovery. The gate only counts ankles; this set is what closure can use.
SUPPORT_LINKS = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)


def has_collision(link: Any) -> bool:
    """True when :func:`capsule_bottom` can measure this link's lowest collision point."""

    return bool(link.collision_mesh_vertices) or link.collision_box_bounds is not None \
        or link.capsule is not None


def build_plan(settings: dict[str, Any]) -> Any:
    """Rebuild the shipped collision-fitted rig plan, same route as the tests."""

    from sim2sense_fall.humans.assets import load_asset_registry
    from sim2sense_fall.humans.config import load_human_config

    config = load_human_config(settings["rig"])
    registry = load_asset_registry(settings["assets"], project_root=REPO_ROOT)
    body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
    mesh = body.model.mesh()
    rest = fit_rest_skeleton(config, mesh.rest_skeleton())
    plan = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
    fit = settings.get("collision_fit", {})
    plan, _ = fit_collision_capsules(
        plan, mesh, **{key: value for key, value in fit.items() if key != "enabled"}
    )
    return plan


def reference_side(settings: dict[str, Any], *, dt_s: float) -> dict[str, Any]:
    """Compare the clip's absolute height curve with per-frame contact closure."""

    plan = build_plan(settings)
    clip = load_action_clip(settings["get_up"], plan, dt_s=dt_s)
    all_links = [link.name for link in plan.links if has_collision(link)]
    links = [name for name in SUPPORT_LINKS if name in all_links]
    rows = []
    for frame in range(len(clip.joints)):
        poses = forward_kinematics(
            plan,
            dict(zip(plan.dof_names, clip.joints[frame], strict=True)),
            root_rotation=clip.tilts[frame],
        )
        bottoms = {name: capsule_bottom(plan, poses, name) for name in links}
        every = {name: capsule_bottom(plan, poses, name) for name in all_links}
        support = min(bottoms, key=bottoms.get)
        global_support = min(every, key=every.get)
        closure_height = -bottoms[support]
        closure_all_height = -every[global_support]
        rows.append({
            "time_s": frame * clip.frame_dt_s,
            "clip_height_m": float(clip.heights_m[frame]),
            "closure_height_m": float(closure_height),
            # >0 means the commanded curve holds the lowest support that far
            # above the floor, so nothing under the body can carry it.
            "float_m": float(clip.heights_m[frame] - closure_height),
            "float_all_links_m": float(clip.heights_m[frame] - closure_all_height),
            "support_link": support,
            "global_support_link": global_support,
            "global_closure_height_m": float(closure_all_height),
            "foot_gaps_m": [float(bottoms["left_ankle"]), float(bottoms["right_ankle"])],
        })
    floats = np.array([row["float_m"] for row in rows])
    floats_all = np.array([row["float_all_links_m"] for row in rows])
    return {
        "clip": clip.provenance,
        "frame_dt_s": clip.frame_dt_s,
        "frames": len(rows),
        "float_m_p50_p95_max": [
            float(np.percentile(floats, 50)), float(np.percentile(floats, 95)),
            float(floats.max()),
        ],
        "float_all_links_m_p50_p95_max": [
            float(np.percentile(floats_all, 50)), float(np.percentile(floats_all, 95)),
            float(floats_all.max()),
        ],
        "frames_floating_above_5mm": int((floats > 0.005).sum()),
        "frames_floating_all_links_above_5mm": int((floats_all > 0.005).sum()),
        "global_support_link_counts": {
            name: int(sum(row["global_support_link"] == name for row in rows))
            for name in sorted({row["global_support_link"] for row in rows})
        },
        "support_link_counts": {
            name: int(sum(row["support_link"] == name for row in rows)) for name in links
        },
        "timeline": rows,
    }


def measured_side(artifact: Path, *, mass_kg: float, gravity_m_s2: float) -> dict[str, Any]:
    """Attribute floor contacts to links and time the unsupported stretches."""

    control = np.load(artifact / "control.npz")
    modes = np.array([str(value) for value in control["mode"]])
    times = control["time_s"]
    contacts = json.loads((artifact / "contacts.json").read_text(encoding="utf-8"))
    if len(contacts) != len(times):
        raise ValueError(f"{artifact}: contact frames {len(contacts)} != control {len(times)}")
    per_frame = []
    for row, mode in zip(contacts, modes, strict=True):
        parts: dict[str, float] = {}
        for sample in row["samples"]:
            if not any(path.endswith("/floor") for path in
                       (sample["collider0_path"], sample["collider1_path"])):
                continue
            path = sample["collider0_path"] if sample["collider0_path"].startswith(
                "/World/Human/") else sample["collider1_path"]
            link = path.removeprefix("/World/Human/").split("/")[0]
            parts[link] = parts.get(link, 0.0) + float(sample["impulse_magnitude_ns"])
        per_frame.append({"mode": str(mode), "parts": parts})
    summary: dict[str, Any] = {}
    dt_s = float(np.median(np.diff(times)))
    for mode in sorted({row["mode"] for row in per_frame}):
        offset = next(index for index, row in enumerate(per_frame) if row["mode"] == mode)
        rows = [row for row in per_frame if row["mode"] == mode]
        touching = [row for row in rows if row["parts"]]
        link_frames: dict[str, int] = {}
        for row in touching:
            for link in row["parts"]:
                link_frames[link] = link_frames.get(link, 0) + 1
        unsupported = np.array([not row["parts"] for row in rows])
        summary[mode] = {
            "frames": len(rows),
            "frames_with_any_floor_contact": int(len(touching)),
            "unsupported_fraction": float(unsupported.mean()),
            "longest_unsupported_s": longest_true_run(unsupported) * dt_s,
            "first_frame_offset_in_mode": offset,
            "link_contact_frames": dict(sorted(link_frames.items(), key=lambda kv: -kv[1])),
        }
    recording = np.load(artifact / "recording.npz")
    skin_min = recording["mesh_vertices_xyz"].min(axis=1)[:, 2]
    indices = np.clip(np.searchsorted(times, recording["time_s"], side="right") - 1,
                      0, len(times) - 1)
    per_skin = []
    for frame, index in enumerate(indices):
        mode = str(modes[index])
        if mode in ("getting_up", "standing_up", "stand", "fallen"):
            per_skin.append({"mode": mode, "skin_min_z_m": float(skin_min[frame])})
    skin_summary = {
        mode: {
            "frames": int(sum(row["mode"] == mode for row in per_skin)),
            "skin_min_z_p50_p95_m": [
                float(np.percentile([r["skin_min_z_m"] for r in per_skin if r["mode"] == mode], q))
                for q in (50, 95)
            ],
            "frames_above_20mm": int(
                sum(r["skin_min_z_m"] > 0.02 for r in per_skin if r["mode"] == mode)
            ),
        }
        for mode in sorted({row["mode"] for row in per_skin})
    }
    return {"per_mode": summary, "skin_by_mode": skin_summary,
            "mass_kg": mass_kg, "gravity_m_s2": gravity_m_s2,
            "force_z_over_weight": force_per_mode(control, modes, mass_kg, gravity_m_s2)}


def longest_true_run(mask: np.ndarray) -> float:
    """Longest contiguous run of True, in frames."""

    best = current = 0.0
    for value in mask:
        current = current + 1.0 if value else 0.0
        best = max(best, current)
    return float(best)


def force_per_mode(control: Any, modes: np.ndarray, mass_kg: float,
                   gravity_m_s2: float) -> dict[str, Any]:
    weight = mass_kg * gravity_m_s2
    out = {}
    for mode in sorted(set(str(value) for value in modes)):
        force = control["force"][modes == mode, 2]
        if not len(force):
            continue
        out[mode] = {
            "p50_of_weight": float(np.median(force) / weight),
            "p95_of_weight": float(np.percentile(force, 95) / weight),
            "frames_at_or_above_0p95W": int((force >= 0.95 * weight).sum()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=None,
                        help="keyboard session directory to analyse (measured side)")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts/humans/getup_float")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    settings = load_keyboard_config(args.config, REPO_ROOT)
    from sim2sense_fall.humans.config import load_human_config

    human = load_human_config(settings["rig"])
    payload: dict[str, Any] = {
        "config": str(args.config.relative_to(REPO_ROOT)),
        "reference_side": reference_side(settings, dt_s=human.simulation.physics_dt_s),
    }
    if args.artifact is not None:
        artifact = args.artifact if args.artifact.is_absolute() else (REPO_ROOT / args.artifact)
        if not artifact.is_dir():
            raise FileNotFoundError(f"no session artifact at {artifact}")
        payload["measured_side"] = measured_side(
            artifact, mass_kg=human.skeleton.mass_kg, gravity_m_s2=human.simulation.gravity_m_s2
        )
    path = args.out / "getup_float.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    reference = payload["reference_side"]
    print(f"reference: {reference['frames']} frames, float_m p50/p95/max "
          f"{reference['float_m_p50_p95_max']}, frames above 5 mm "
          f"{reference['frames_floating_above_5mm']}")
    print(f"support links: {reference['support_link_counts']}")
    if "measured_side" in payload:
        for mode, row in payload["measured_side"]["per_mode"].items():
            print(f"measured {mode:12s} frames={row['frames']:5d} "
                  f"unsupported={row['unsupported_fraction']:.2f} "
                  f"longest_unsupported_frames={row['longest_unsupported_s']:.0f} "
                  f"top_links={list(row['link_contact_frames'].items())[:4]}")
        for mode, row in payload["measured_side"]["force_z_over_weight"].items():
            print(f"force_z/W {mode:12s} p50={row['p50_of_weight']:.2f} "
                  f"p95={row['p95_of_weight']:.2f} at_cap={row['frames_at_or_above_0p95W']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
