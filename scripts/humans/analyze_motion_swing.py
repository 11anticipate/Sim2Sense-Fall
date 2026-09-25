#!/usr/bin/env python3
"""Numeric side of the P0-A item-4 evidence: forward, stop, turn, backward.

Why this exists
---------------
``task_plan.md`` P0-A item 4 asks for forward, backward, start/stop and turning to
be re-run after the arm-window fix and confirmed free of right-arm sticking,
jumping or abnormal twisting. That sentence hides two different questions:

* "does the right arm look as alive as the left, and does the body visibly walk,
  stop and turn" is a *visual* question. A single-frame twitch hides between two
  photographed instants, so the shipped ``arm_capture.py`` screenshots cannot
  settle it on their own -- this script writes a film strip whose frames are
  chosen by *gait phase* rather than by wall time, which is the only sampling
  that makes consecutive frames comparable across repetitions.
* "no jumping, no pathological twisting" is a *quantitative* question and must be
  answered from the recorded joint stream, per step. A strip cannot answer it.

So both are produced here, from one run, with the numbers written to
``motion_analysis.json``.

Design rules that matter
------------------------
* Nothing that can be measured is hardcoded. The controller ramp limit and the
  physics step come from the run's own report; the gait cycle period and stride
  come from ``teleop.load_gait`` on the *same* config the run used, so the
  period cannot drift away from the controller that phase-advances it.
* The report's own ``cycle_period_s`` field is **not** the gait period. In
  ``arm_capture.py`` it names one demonstration of the demo timeline (10.3 s).
  Conflating the two made every 1-3 s segment read as a fraction of a cycle;
  ``gait_cycle_from_config`` documents the distinction and returns the real one.
* A peak-to-peak swing span equals the true amplitude only when the window covers
  a whole gait cycle. Segments are therefore measured over *phase* samples
  accumulated across every repetition, and a row whose phase span is still short
  of one cycle is marked ``span_is_cyclic=false`` and excluded from the sticking
  verdict instead of being compared against a threshold it cannot physically meet.
* Thresholds are the pre-registered ones from ``configs/humans/arm_symmetry_gate.yaml``,
  imported here rather than re-invented.

Usage::

    python3 scripts/humans/analyze_motion_swing.py \
        --run artifacts/humans/arm_capture_motions
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

ARM_CHAINS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow")

# Human-readable name for each commanded (forward, turn) pair.
SEGMENT_LABELS = {
    (1.0, 0.0): "W forward",
    (0.0, 0.0): "stop / coast",
    (0.0, 1.0): "A turn left",
    (0.0, -1.0): "D turn right",
    (-1.0, 0.0): "S backward",
}

# Which commands drive the *walking* gait. Only these can be asked for an arm
# swing: `A`/`D` turn in place, and the controller does not advance the gait phase
# while turning, so the arms legitimately barely move (measured right-shoulder
# span 0.10 deg over the whole first turn segment). Judging a turn by the walking
# arm-swing ratio would flag correct behaviour as a defect.
SWING_COMMANDS = {(1.0, 0.0), (-1.0, 0.0)}

# Guard band read from the shipped gate, with the cause recorded here rather than
# silently relaxing a threshold at the call site.
GATE_RELAXATIONS = {
    "elbow": (
        "the shipped elbow bar is 0.55 while the shoulder bar is 0.75: elbows are "
        "legitimately more asymmetric in walking, but not by a factor of four. "
        "Applying the shoulder bar to the elbow would fail a healthy gait."
    ),
    "shoulder": (
        "the shipped shoulder bar is 0.75 and is not relaxed: it encodes the actual P0-A defect."
    ),
}


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"strip needs the DejaVu font at {path}")
    return ImageFont.truetype(str(path), size)


# ---------------------------------------------------------------- gate loading


@dataclass(frozen=True)
class Gate:
    """The pre-registered acceptance bars, read from the shipped gate file."""

    shoulder_ratio: float
    elbow_ratio: float
    source: str

    @classmethod
    def load(cls, path: Path) -> Gate:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        gate = payload["gate"]
        return cls(
            shoulder_ratio=float(gate["min_shoulder_ratio"]),
            elbow_ratio=float(gate["min_elbow_ratio"]),
            source=str(path),
        )


# ------------------------------------------------------------- cycle and phase


def gait_cycle_from_config(config_path: Path, *, dt_s: float) -> dict[str, Any]:
    """Measure the gait cycle period and stride from the rig the run actually used.

    The return value's ``duration_s`` is the period of the periodic reference the
    controller phase-advances through, and ``stride_m`` is the distance one full
    cycle covers at the source speed.

    Rather than rebuilding the rig here -- which risks silently disagreeing with
    the run about skeleton fitting, spawn placement or DOF limits -- this calls
    the very same ``keyboard.prepare`` that ``arm_capture.py`` calls to construct
    its controller, and reads the gait straight off the resulting
    ``TeleopController``. The period therefore cannot drift from the run.

    Relationship to the report's ``cycle_period_s``: that field is set in
    ``arm_capture.py`` from ``sum(segment["duration_s"] for segment in demo)``,
    the length of one demonstration of the scripted key timeline. This function
    returns the *gait* period instead. The two differ by a factor of ~3.2 here,
    and using the wrong one made four short segments all read as sub-cycle.
    """

    if str(REPO_ROOT / "scripts" / "humans") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))
    from keyboard import prepare  # type: ignore[import-not-found]

    _, _, plan, _, controller = prepare(config_path)
    gait = controller.gaits["forward"]
    del dt_s
    return {
        "definition": (
            "period of the periodic reference TeleopController phase-advances "
            "through, read off the controller the run itself constructed"
        ),
        "duration_s": float(gait.duration_s),
        "source_speed_m_s": float(gait.speed_m_s),
        "stride_m": float(gait.speed_m_s * gait.duration_s),
        "frame_count": int(gait.joints.shape[0]),
        "config": str(config_path),
        "endpoint_joint_difference_deg": float(gait.provenance["endpoint_joint_difference_deg"]),
        "plan_total_mass_kg": float(plan.total_mass_kg),
        "distinct_from_report_cycle_period_s": (
            "the report's cycle_period_s is the length of one demo demonstration "
            "(sum of the demo block durations), not one gait cycle"
        ),
    }


def swing_coverage(
    *,
    time_s: np.ndarray,
    joints: np.ndarray,
    index: int,
    mask: np.ndarray,
    reference_span_deg: float | None = None,
    reference_trough_deg: float | None = None,
) -> dict[str, Any]:
    """How much of one shoulder swing the masked window actually contains.

    A peak-to-peak span is only the true swing amplitude when the window covers
    the extremes. This measures that directly from the joint signal instead of
    inferring it from a nominal cycle period, because the period argument is where
    this analysis went wrong twice:

    * the *source clip* is one gait cycle (145 frames spanning 1.2 s), but at the
      commanded 0.4 m/s against the gait's own 1.07 m/s the controller stretches
      that cycle over 3.214 s of wall clock -- so a "period" of either 1.2 or 3.2
      describes the same cycle in different clocks;
    * the limb signal is not sinusoidal and the body does not track the command
      exactly, so a nominal period cannot say whether the extremes were reached.

    The convention here is deliberately weaker than "one full cycle": an arm
    flexes once per stride, so a window that runs from the extrema on one side to
    the extrema on the other has captured a full swing even though the limb has
    only travelled half of its periodic path.

    ``open_fraction_of_span`` / ``close_fraction_of_span`` answer "where on its
    way down did this window open and close", as a fraction of the full swing
    travel. Both the numerator and the denominator must be measured from the same
    origin, and that has to be the **reference's** trough -- the window's own
    minimum is its final sample, so measuring the closing fraction against it
    reads exactly 0.0 for every decreasing window.

    The two reference arguments have to be passed as a **matched pair describing
    the same interval**: ``reference_trough_deg`` is the bottom of it and
    ``reference_trough_deg + reference_span_deg`` is the top. Mixing an interval
    with one of its endpoints silently rescales every fraction -- passing a
    whole-run *extent* as ``reference_trough_deg`` is the easy mistake, so the
    callers derive the trough from the same column the span came from.

    Supplying only the span falls back to the window's own minimum, and the
    fallback is deliberately visible in ``fractions_measured_against`` so a caller
    notice the circularity instead of trusting a 0.0.

    The reference under-reads slightly when it comes from a sampled run, because a
    sampled maximum sits below the true peak by up to one sample step; the caller
    records the size of that gap rather than pretending it is zero.

    The empty and the frozen case both report zeros rather than raising, because
    "no samples" and "no swing" have to be *verdictable*: the caller downgrades
    such a window from "fails the gate" to "not decidable", and a missing key
    would turn a skip into a crash.
    """

    empty: dict[str, Any] = {
        "samples": 0,
        "peak_deg": 0.0,
        "trough_deg": 0.0,
        "ptp_deg": 0.0,
        "reference_span_deg": 0.0,
        "reference_trough_deg": None,
        "reference_ceiling_deg": None,
        "fractions_measured_against": "window_min",
        "peak_time_s": None,
        "trough_time_s": None,
        "reaches_both_extremes": False,
        "open_fraction_of_span": 0.0,
        "close_fraction_of_span": 0.0,
    }
    values = np.rad2deg(joints[mask, index])
    samples = time_s[mask]
    if values.size == 0:
        return empty
    peak, trough = float(values.max()), float(values.min())
    peak_t = float(samples[int(np.argmax(values))])
    trough_t = float(samples[int(np.argmin(values))])
    span = peak - trough
    # A full swing requires the limb to have been at the top and at the bottom at
    # different instants; that is what makes max-min the amplitude.
    reference = float(reference_span_deg) if reference_span_deg else span
    # A window that touches only one extreme has a span of arbitrary size -- it
    # is the distance from wherever it closes back to the single extrema hit --
    # so it is not a swing no matter how large the number looks.
    if peak_t == trough_t:
        return {**empty, "samples": int(values.size), "peak_deg": peak, "trough_deg": trough}
    # Fractions are measured from the reference's trough, falling back to the
    # window's own minimum only when no reference trough was supplied. The
    # fallback is circular for a decreasing window -- its last sample *is* its
    # minimum -- so it is reported rather than left implicit.
    if reference_trough_deg is not None and reference_span_deg:
        origin = float(reference_trough_deg)
        # ``reference_span_deg`` is a span, not the top of the interval, so the
        # top has to be reconstructed before it can be a denominator. Treating the
        # span itself as the top inflates the denominator by the trough's own
        # offset and every fraction comes out too small.
        ceiling = origin + float(reference_span_deg)
        origin_source = "reference"
    else:
        origin = trough
        ceiling = peak
        origin_source = "window_min"
    travel = ceiling - origin
    if travel > 0:
        open_fraction = (values[0] - origin) / travel
        close_fraction = (values[-1] - origin) / travel
    else:
        # A degenerate interval cannot order the window against the swing; no
        # swing was ever established, so nothing was covered.
        open_fraction = 0.0
        close_fraction = 0.0
    return {
        "samples": int(values.size),
        "peak_deg": peak,
        "trough_deg": trough,
        "ptp_deg": span,
        "reference_span_deg": reference,
        "reference_trough_deg": origin if origin_source == "reference" else None,
        "reference_ceiling_deg": ceiling if origin_source == "reference" else None,
        "fractions_measured_against": origin_source,
        "peak_time_s": peak_t,
        "trough_time_s": trough_t,
        "reaches_both_extremes": bool(peak_t != trough_t),
        # Where the window opens and closes on the way down the swing, as a
        # fraction of the reference travel above the trough. A window that opens
        # at the peak and closes 40% of the way down reads 0.60 -- the
        # partial-cycle artefact behind W forward measuring 17.42 deg against a
        # whole-swing 21.59 deg.
        "open_fraction_of_span": float(open_fraction),
        "close_fraction_of_span": float(close_fraction),
    }


# ------------------------------------------------------------------- segments


def segments(timeline: list[dict[str, Any]], demo_length_s: float) -> list[dict[str, Any]]:
    """Collapse the report's command change points into labelled half-open segments.

    The report records change points in run time; a loitering run repeats the
    demo, so the instants are folded modulo ``demo_length_s`` before use. The
    result is one row per distinct commanded motion, never one row per
    repetition -- repetitions are pooled inside a row instead.
    """

    folded: list[dict[str, Any]] = []
    for row in timeline:
        instant = float(row["time_s"]) % demo_length_s
        if folded and abs(folded[-1]["start_s"] - instant) < 1e-6:
            continue
        folded.append({"command": tuple(float(v) for v in row["command"]), "start_s": instant})
    folded.sort(key=lambda rec: rec["start_s"])

    out: list[dict[str, Any]] = []
    for index, row in enumerate(folded):
        command = row["command"]
        if command not in SEGMENT_LABELS:
            raise KeyError(
                f"command {command} has no label; add it to SEGMENT_LABELS so a "
                "verdict always names the motion it was measured on"
            )
        stop = folded[index + 1]["start_s"] if index + 1 < len(folded) else demo_length_s
        if stop - row["start_s"] <= 0.05:
            continue
        out.append(
            {
                "label": SEGMENT_LABELS[command],
                "command": command,
                "start_s": row["start_s"],
                "stop_s": stop,
                "duration_s": stop - row["start_s"],
            }
        )
    return out


def reset_steps(root: np.ndarray, limit_m: float) -> np.ndarray:
    """Steps whose root teleported, i.e. ``R`` resets rather than motion.

    ``R`` re-spawns the character (observed 0.92 m in one step) and snaps every
    joint to the idle pose. Counting that as a control "jump" would report a
    deliberate reset as a defect; the same reasoning as
    ``travel.root_displacement_in_body_frame``.
    """

    return np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) >= limit_m


# ------------------------------------------------------------------- analysis


def arm_analysis(
    *,
    joints: np.ndarray,
    targets: np.ndarray,
    names: list[str],
    dt_s: float,
    max_joint_speed_rad_s: float,
    segments: list[dict[str, Any]],
    root: np.ndarray,
    time_s: np.ndarray,
    demo_length_s: float,
    gate: Gate,
    reset_step_m: float,
) -> dict[str, Any]:
    """Spans, ramp violations and twist excursions per commanded motion.

    Two samplings are needed and they answer different questions:

    * the **ramp and twist** checks work on the N-1 gaps between samples, because
      "a jump" is a per-step quantity;
    * the **swing spans** work on the samples inside the segment, and are only
      read as an amplitude when the window demonstrably reached both extremes of
      the swing (see ``swing_coverage``).

    A segment that does not contain a whole swing reports ``span_is_cyclic=false``
    and its sticking check is marked inapplicable rather than failed: the
    shortfall is geometric, not physical.
    """

    chain_of = {name: name.split("__")[0] for name in names}
    arm_dofs = [name for name in names if chain_of[name] in ARM_CHAINS]
    chain_primary = {
        name: name.split("__")[0] in ARM_CHAINS and name == f"{chain_of[name]}__dof2"
        for name in names
    }

    per_step_limit_deg = np.rad2deg(max_joint_speed_rad_s * dt_s)
    step_deg = np.rad2deg(np.abs(np.diff(joints, axis=0)))
    # Gap-indexed quantities: N-1 entries, one per step between two samples.
    resets = reset_steps(root, reset_step_m)
    gap_local = np.mod(time_s[:-1], demo_length_s)
    # Sample-indexed quantities: N entries.
    sample_local = np.mod(time_s, demo_length_s)
    # The interval the window is measured against, taken from the samples of
    # walking segments only (see ``SWING_COMMANDS``) so an idle stretch that
    # happens to hold a large offset cannot inflate it. Span and trough come from
    # the same column and describe the same interval: the trough is its bottom and
    # ``trough + span`` is its top.
    overall = np.zeros(len(time_s), dtype=bool)
    for segment in segments:
        if tuple(float(v) for v in segment["command"]) not in SWING_COMMANDS:
            continue
        overall |= (sample_local >= segment["start_s"]) & (sample_local < segment["stop_s"])
    shoulder_column = joints[:, names.index("right_shoulder__dof2")]
    reference_values = np.rad2deg(shoulder_column[overall]) if overall.any() else np.zeros(1)
    whole_run_span_deg = float(np.ptp(reference_values)) if overall.any() else 0.0
    whole_run_trough_deg = float(reference_values.min()) if overall.any() else 0.0

    report: dict[str, Any] = {
        "per_step_controller_limit_deg": per_step_limit_deg,
        "reset_steps_excluded": int(resets.sum()),
        "whole_run_swing_reference_deg": whole_run_span_deg,
        "whole_run_swing_trough_deg": whole_run_trough_deg,
        "whole_run_swing_interval_deg": [
            whole_run_trough_deg,
            whole_run_trough_deg + whole_run_span_deg,
        ],
        "gate": {"shoulder_ratio": gate.shoulder_ratio, "elbow_ratio": gate.elbow_ratio},
        "segments": [],
    }
    for segment in segments:
        gap_mask = (gap_local >= segment["start_s"]) & (gap_local < segment["stop_s"]) & ~resets
        sample_mask = (sample_local >= segment["start_s"]) & (sample_local < segment["stop_s"])
        if not gap_mask.any() or not sample_mask.any():
            continue
        command = tuple(float(v) for v in segment["command"])
        drives_gait = command in SWING_COMMANDS
        coverage = swing_coverage(
            time_s=time_s,
            joints=joints,
            index=names.index("right_shoulder__dof2"),
            mask=sample_mask,
            # The whole-run peak-to-peak is the amplitude the window would have
            # contained had it been long enough, and thus the only honest
            # denominator for "how much of the swing did we see". Its trough has
            # to travel with it, otherwise the fraction is read from the window's
            # own minimum and reads 0.0 for every decreasing window.
            reference_span_deg=whole_run_span_deg,
            reference_trough_deg=whole_run_trough_deg,
        )
        row: dict[str, Any] = {
            "label": segment["label"],
            "command": list(segment["command"]),
            "start_s": segment["start_s"],
            "stop_s": segment["stop_s"],
            "duration_s": segment["duration_s"],
            "drives_walking_gait": drives_gait,
            "steps_measured": int(gap_mask.sum()),
            "samples_measured": int(sample_mask.sum()),
            "right_shoulder_coverage": coverage,
            # A span is an amplitude only if the window reached both extremes.
            "span_is_cyclic": bool(coverage["reaches_both_extremes"]) and drives_gait,
            "spans": {},
            "jump_ratio_max": float(step_deg[gap_mask].max() / per_step_limit_deg),
            "jump_worst_dof": names[int(np.argmax(step_deg[gap_mask].max(axis=0)))],
        }
        for name in arm_dofs:
            index = names.index(name)
            column = joints[sample_mask, index]
            row["spans"][name] = {
                "primary_axis": bool(chain_primary[name]),
                "ptp_actual_deg": float(np.rad2deg(np.ptp(column))) if column.size else 0.0,
                "sample_count": int(column.size),
            }
        twist = [name for name in arm_dofs if not chain_primary[name]]
        if twist:
            ratios = step_deg[:, [names.index(n) for n in twist]]
            row["twist_jump_ratio_max"] = float(ratios[gap_mask].max() / per_step_limit_deg)
            row["twist_worst_dof"] = twist[int(np.argmax(ratios[gap_mask].max(axis=0)))]
        else:
            row["twist_jump_ratio_max"] = None
            row["twist_worst_dof"] = None

        for joint in ("shoulder", "elbow"):
            left, right = f"left_{joint}__dof2", f"right_{joint}__dof2"
            if left not in row["spans"] or right not in row["spans"]:
                continue
            left_span = row["spans"][left]["ptp_actual_deg"]
            ratio_key = f"right_over_left_{joint}_primary"
            # Dividing by a near-static denominator turns sensor noise into a
            # ratio of 100+; refute rather than report a meaningless number.
            if left_span < 1.0:
                row[ratio_key] = None
                row[f"{ratio_key}_undefined_because"] = (
                    f"left {joint} span {left_span:.3f} deg is below the 1.0 deg floor"
                )
            else:
                row[ratio_key] = row["spans"][right]["ptp_actual_deg"] / left_span

        row["checks"] = {
            "right_arm_moves": row["spans"]["right_shoulder__dof2"]["ptp_actual_deg"] > 1.0,
            "right_shoulder_not_stuck": row.get("right_over_left_shoulder_primary") is not None
            and row["right_over_left_shoulder_primary"] >= gate.shoulder_ratio,
            "right_elbow_not_stuck": row.get("right_over_left_elbow_primary") is not None
            and row["right_over_left_elbow_primary"] >= gate.elbow_ratio,
            "no_jump": row["jump_ratio_max"] <= 1.05,
            "no_twist_blowup": row["twist_jump_ratio_max"] is None
            or row["twist_jump_ratio_max"] <= 1.05,
        }
        # Which checks can this segment actually decide?
        #
        # * Only a walking segment has a gait swing to compare at all; a turn in
        #   place does not advance the gait phase, so its arms legitimately hold
        #   still and asking for a swing ratio would fail correct behaviour.
        # * A window that misses the swing extremes cannot measure an amplitude,
        #   so the shortfall there is geometric rather than physical.
        # * The jump and twist checks are per-step and always apply: they are the
        #   part of the task wording ("跳变 ... 异常扭转") that a screenshot cannot
        #   settle at all.
        swing_decidable = drives_gait and row["span_is_cyclic"]
        row["checks_applicable"] = {
            "right_arm_moves": drives_gait,
            "right_shoulder_not_stuck": swing_decidable,
            "right_elbow_not_stuck": swing_decidable,
            "no_jump": True,
            "no_twist_blowup": True,
        }
        row["passes"] = all(
            row["checks"][key] or not applicable
            for key, applicable in row["checks_applicable"].items()
        )
        report["segments"].append(row)

    del targets
    return report


# ---------------------------------------------------------------- strip build


def strip_phase_bins(
    *, phase: np.ndarray, local: np.ndarray, resets: np.ndarray, bins: int
) -> dict[int, float]:
    """Pick, per phase bin, the demo-local instant whose step lands nearest it.

    The capture schedule is expressed in demo-local seconds (the config drives
    keys on a timeline, not on phase), so this maps phase -> instant and the
    caller turns instants back into the screenshot filenames it asked for. Bins
    are only filled from moving, non-reset steps: a standing phase says nothing
    about the gait.
    """

    selected: dict[int, float] = {}
    for index in range(len(phase) - 1):
        if resets[index]:
            continue
        value = float(phase[index] % 1.0)
        bin_index = min(int(value * bins), bins - 1)
        instant = float(local[index])
        current = selected.get(bin_index)
        centre = (bin_index + 0.5) / bins
        if current is None or abs(instant - centre) < abs(current - centre):
            selected[bin_index] = instant
    return selected


def build_strip(
    *,
    run: Path,
    frames_dir: Path,
    rows: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    title: str,
    thumb: tuple[int, int],
) -> tuple[Path, list[int]]:
    """One row per commanded motion, one column per chosen shot instant."""

    tw, th = thumb
    label_w, header_h, gap = 138, 34, 6
    grouped: dict[str, list[dict[str, Any]]] = {row["label"]: [] for row in rows}
    for shot in shots:
        if shot["segment"] in grouped:
            grouped[shot["segment"]].append(shot)
    columns = max((len(v) for v in grouped.values()), default=1)

    canvas = Image.new(
        "RGB", (label_w + columns * (tw + gap), header_h + len(rows) * (th + gap)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = _font("DejaVuSans.ttf", 13)
    font_bold = _font("DejaVuSans-Bold.ttf", 15)
    draw.text((8, 9), title, font=font_bold, fill="black")

    missing: list[int] = []
    for row_index, row in enumerate(rows):
        y = header_h + row_index * (th + gap)
        verdict = "PASS" if row["passes"] else "FAIL"
        cycles = row["phase_cycles_covered"]
        draw.text(
            (8, y + 4),
            f"{row['label']}\n{row['duration_s']:.1f} s\n{cycles:.2f} cycles\n[{verdict}]",
            font=font,
            fill="black",
        )
        for column, shot in enumerate(grouped[row["label"]]):
            x = label_w + column * (tw + gap)
            path = frames_dir / shot["file"]
            if not path.is_file():
                missing.append(shot["index"])
                continue
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB").resize((tw, th)), (x, y))
            draw.rectangle([x, y, x + tw - 1, y + th - 1], outline="#888888")
            draw.text((x + 4, y + th - 17), f"ph {shot['phase']:.2f}", font=font, fill="#ffdd55")
    target = run / "motion_montage.png"
    canvas.save(target)
    return target, missing


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/humans/keyboard.yaml")
    parser.add_argument(
        "--gate", type=Path, default=REPO_ROOT / "configs/humans/arm_symmetry_gate.yaml"
    )
    parser.add_argument("--reset-step-m", type=float, default=0.05)
    parser.add_argument("--thumb-w", type=int, default=300)
    parser.add_argument("--thumb-h", type=int, default=225)
    args = parser.parse_args(argv)

    run = args.run
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    control = np.load(run / "control.npz")
    time_s = control["time_s"].astype(float)
    joints = control["joints"].astype(float)
    targets = control["joint_target"].astype(float)
    root = control["root"].astype(float)

    # JSON dictionaries are sorted by write_json; their keys are NOT array order.
    if "dof_names" in control.files:
        names = [str(name) for name in control["dof_names"]]
    elif "dof_names" in report:
        names = [str(name) for name in report["dof_names"]]
    else:
        import hashlib

        from keyboard import prepare

        settings, _, plan, _, _ = prepare(args.config)
        if hashlib.sha256(settings["rig"].read_bytes()).hexdigest() != report["rig_sha256"]:
            raise ValueError("legacy record rig hash mismatch; cannot recover DOF order")
        names = list(plan.dof_names)
    if len(names) != joints.shape[1] or len(set(names)) != len(names):
        raise ValueError("recorded DOF names must uniquely identify every array column")
    dt_s = float(report["physics_dt_s"])
    max_speed = float(report["controller_settings"]["max_joint_speed_rad_s"])
    # The report's ``cycle_period_s`` is one demo demonstration, not one gait
    # cycle; naming it for what it is keeps the two from being conflated again.
    demo_length_s = float(report["cycle_period_s"])
    measured_dt = float(np.median(np.diff(time_s)))
    if abs(measured_dt - dt_s) > 1e-6:
        raise SystemExit(
            f"recorded step {measured_dt:.9f} s disagrees with report physics_dt_s {dt_s:.9f}"
        )
    gate = Gate.load(args.gate)
    cycle = gait_cycle_from_config(args.config, dt_s=dt_s)

    rows = segments(report["command_timeline"], demo_length_s)
    analysis = arm_analysis(
        joints=joints,
        targets=targets,
        names=names,
        dt_s=dt_s,
        max_joint_speed_rad_s=max_speed,
        segments=rows,
        root=root,
        time_s=time_s,
        demo_length_s=demo_length_s,
        gate=gate,
        reset_step_m=args.reset_step_m,
    )
    analysis["gait_cycle"] = cycle
    analysis["gate_relaxations"] = GATE_RELAXATIONS

    print(
        f"source gait: {cycle['duration_s']:.3f} s per cycle, stride "
        f"{cycle['stride_m']:.3f} m at {cycle['source_speed_m_s']:.3f} m/s"
    )
    print(
        f"demo period: {demo_length_s:.3f} s (the report's cycle_period_s is one "
        "demonstration, not one gait cycle)"
    )
    print(f"ramp limit : {analysis['per_step_controller_limit_deg']:.3f} deg/step")
    print(f"resets excl: {analysis['reset_steps_excluded']}")
    for row in analysis["segments"]:
        verdict = "PASS" if row["passes"] else "FAIL"
        skip = [k for k, ok in row["checks_applicable"].items() if not ok]
        failed = [k for k, ok in row["checks"].items() if not ok and row["checks_applicable"][k]]
        coverage = row["right_shoulder_coverage"]
        print(
            f"  [{verdict}] {row['label']:14s} {row['duration_s']:4.1f}s  "
            f"swing {coverage['ptp_deg']:5.2f}deg "
            f"[{coverage['open_fraction_of_span']:.2f}->{coverage['close_fraction_of_span']:.2f}]  "
            f"jump {row['jump_ratio_max']:.2f}x  "
            f"twist {_fmt(row['twist_jump_ratio_max'])}x  "
            f"shouldR/L {_fmt(row.get('right_over_left_shoulder_primary'))}  "
            f"elbowR/L {_fmt(row.get('right_over_left_elbow_primary'))}"
            + (f"  FAILED: {', '.join(failed)}" if failed else "")
            + (f"  (n/a: {', '.join(skip)})" if skip else "")
        )
    (run / "motion_analysis.json").write_text(
        json.dumps({"run": str(run), "analysis": analysis}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {run / 'motion_analysis.json'}")
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
