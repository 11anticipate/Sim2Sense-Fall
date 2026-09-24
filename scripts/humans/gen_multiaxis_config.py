"""Generate configs/humans/human_smpl_multiaxis.yaml from the neutral base config.

Kept as a script so the joints block has one authoritative definition: the axis
order, the limits and the gains are all here, and the prose in the generated file
records why each choice was made.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path(".")
BASE = Path("configs/humans/human_smpl_neutral.yaml")
TARGET = Path("configs/humans/human_smpl_multiaxis.yaml")

# (rotation order as declared, limits per axis in that order, drive).
# Orders put the PRIMARY axis last so the real link keeps the joint's historical
# axis and name. Limits come from scripts/humans/measure_axes.py, which reports the
# angle spread of a 40-clip AMASS sample; each was widened until the sample stopped
# clipping, because a limit narrower than the motion replays a different pose.
JOINTS: dict[str, tuple[tuple[str, ...], list[tuple[int, int]], tuple[int, int, int]]] = {
    "left_hip":      (("x", "z", "y"), [(-40, 40), (-45, 45), (-110, 45)], (1200, 120, 400)),
    "right_hip":     (("x", "z", "y"), [(-40, 40), (-30, 30), (-105, 45)], (1200, 120, 400)),
    "left_knee":     (("x", "z", "y"), [(-30, 30), (-60, 60), (-25, 120)], (900, 90, 300)),
    "right_knee":    (("x", "z", "y"), [(-30, 30), (-40, 40), (-20, 120)], (900, 90, 300)),
    "left_ankle":    (("x", "z", "y"), [(-30, 30), (-40, 40), (-55, 35)], (450, 45, 150)),
    "right_ankle":   (("x", "z", "y"), [(-30, 30), (-35, 35), (-75, 55)], (450, 45, 150)),
    "spine1":        (("x", "z", "y"), [(-30, 30), (-35, 35), (-30, 90)], (1500, 150, 500)),
    "spine2":        (("x", "z", "y"), [(-30, 30), (-30, 30), (-30, 50)], (1500, 150, 500)),
    "spine3":        (("x", "z", "y"), [(-25, 25), (-25, 25), (-30, 35)], (1500, 150, 500)),
    "neck":          (("x", "z", "y"), [(-35, 35), (-55, 55), (-45, 50)], (360, 36, 120)),
    "head":          (("x", "z", "y"), [(-35, 35), (-45, 45), (-55, 45)], (360, 36, 120)),
    "left_collar":   (("x", "z", "y"), [(-55, 55), (-75, 75), (-50, 35)], (450, 45, 150)),
    "right_collar":  (("x", "z", "y"), [(-55, 55), (-75, 75), (-50, 35)], (450, 45, 150)),
    "left_shoulder": (("y", "z", "x"), [(-70, 70), (-65, 65), (-125, 40)], (600, 60, 200)),
    "right_shoulder": (("y", "z", "x"), [(-70, 70), (-65, 65), (-40, 120)], (600, 60, 200)),
    "left_elbow":    (("x", "y", "z"), [(-65, 65), (-75, 75), (-170, 30)], (450, 45, 150)),
    "right_elbow":   (("x", "y", "z"), [(-65, 65), (-75, 75), (-25, 155)], (450, 45, 150)),
    "left_wrist":    (("x", "z", "y"), [(-105, 105), (-60, 60), (-95, 45)], (300, 30, 120)),
    "right_wrist":   (("x", "z", "y"), [(-105, 105), (-60, 60), (-105, 65)], (300, 30, 120)),
}

HEADER = """  # Every joint that recorded human motion rotates is driven by a chain of three
  # revolute joints, one per body axis, so the rig can reproduce an arbitrary joint
  # rotation instead of only the dominant sagittal one. Measured, not assumed:
  # scripts/humans/measure_axes.py reports the single-axis rig reproduces 0.0000% of
  # sampled AMASS frames, with median misses of 7-26 degrees and p99.5 misses up to
  # 70 degrees at the shoulder and elbow. Two axes are not enough either -- they still
  # miss 1.6-12 degrees -- because only three axes span the whole rotation group.
  #
  # Chain order puts the PRIMARY axis last, so the real link carries the axis the
  # single-axis rig used. Any order spans the same rotations, and keeping the
  # historical axis last means the joint name, its anatomical range and any
  # visualization.default_pose_rad entry keep naming the same physical direction.
  #
  # Limits are the measured angle spread with margin, from the same tool that
  # accepted the rig: a limit narrower than the motion clips the replay, and a limit
  # far wider lets the physics fold the joint the wrong way. The secondary axes are
  # sized the same way, because they carry the residue of the Euler split rather than
  # an independent anatomical degree of freedom. scripts/humans/gen_multiaxis_config.py
  # regenerates this block, so the numbers here are not hand-drifted.
  #
  # Hands are not listed: SMPL's hand joints have no AMASS counterpart. Feet are not
  # listed either, and both measure exactly zero rotation in the local corpus.
  # measure_axes.py fails when a corpus rotates a joint the rig does not drive, so
  # that assumption cannot rot silently.
  #
  # A chain splits its segment's mass across three links, and three drives in series
  # are softer than one, so stiffness and damping are scaled up relative to
  # human_smpl_neutral.yaml. The values are a starting point to be judged by measured
  # tracking error, not a claim.
  joints:
"""


def main() -> None:
    text = BASE.read_text()
    entries = []
    for joint, (axes, limits, (stiffness, damping, max_force)) in JOINTS.items():
        rendered = ", ".join(f"[{lo}, {hi}]" for lo, hi in limits)
        entries.append(
            f"    {joint + ':':<15} {{dof: revolute, rotations: [{', '.join(axes)}],\n"
            f"                    limits_deg: [{rendered}],\n"
            f"                    drive: {{stiffness: {stiffness}, damping: {damping},"
            f" max_force: {max_force}}}}}"
        )
    start = text.index("  # Only joints with a degree of freedom are listed.")
    end = text.index("# The imported SMPL neutral asset")
    generated = text[:start] + HEADER + "\n".join(entries) + "\n\n" + text[end:]
    generated = generated.replace(
        "human_id: smpl_neutral_standing", "human_id: smpl_amass_multiaxis", 1
    )
    TARGET.write_text(generated)
    print(f"wrote {TARGET} with {len(JOINTS)} multi-axis joints")


if __name__ == "__main__":
    main()
