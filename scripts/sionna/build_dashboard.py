#!/usr/bin/env python3
"""Build a static sample-browser dashboard for a batch of Sionna CIR samples.

One self-contained HTML file: a summary of the batch (from the step-detector
and DTW reports when they exist), a table of every admitted channel sample,
and per-sample panels — CIR waterfall, range-time map, channel-feature
timelines with the recorded onset/impact markers, and both detectors' frame
scores — embedded as base64 PNG so the file opens offline with a double
click. Runs on the system Python (matplotlib + numpy only); reads artifacts,
never writes into them.

This is a *record browser* for inspection and threshold tuning. It is not an
interactive detector and reports no performance of its own.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2sense_fall.detection import (  # noqa: E402
    StepDetectorConfig,
    cir_features,
    detect_fall,
)
from sim2sense_fall.detection_data import ChannelSample, iter_channel_samples  # noqa: E402
from sim2sense_fall.dtw_baseline import (  # noqa: E402
    TemplateConfig,
    build_fall_template,
    detect_fall_template,
)
from sim2sense_fall.range_time import RangeTimeConfig, range_time_map  # noqa: E402
from sim2sense_fall.windowing import WindowConfig  # noqa: E402


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Knobs of both detectors and the representation, one place."""

    detector: StepDetectorConfig
    window: WindowConfig
    template: TemplateConfig
    range_time: RangeTimeConfig


def default_config() -> DashboardConfig:
    """Defaults matching the evaluate scripts' pre-registered knobs."""

    return DashboardConfig(
        detector=StepDetectorConfig(),
        window=WindowConfig(
            window_length_s=0.2, stride_s=0.1, threshold=0.8,
            min_consecutive_positive_windows=2,
        ),
        template=TemplateConfig(
            resample_points=64, dba_iterations=3, dtw_band_fraction=0.15,
            window_s=0.8,
        ),
        range_time=RangeTimeConfig(),
    )


def _latency(first_alarm_s: float | None, onset_s: float | None) -> float | None:
    if first_alarm_s is None or onset_s is None:
        return None
    return first_alarm_s - onset_s


def sample_view(
    sample: ChannelSample, template: np.ndarray | None, config: DashboardConfig
) -> dict:
    """Both detectors' verdicts and score series for one sample."""

    features = cir_features(sample.cir, sample.delay_s)
    step = detect_fall(features, sample.time_s, config.detector, config.window)
    if template is not None and template.size:
        dtw = detect_fall_template(
            sample.cir, sample.delay_s, sample.time_s, template, config.template,
            config.window,
        )
    else:
        dtw = {
            "frame_scores": np.zeros(len(sample.time_s)),
            "window_alarmed": False,
            "first_alarm_s": None,
        }
    return {
        "sample_id": sample.sample_id,
        "activity": sample.activity,
        "event_label": sample.event_label or "-",
        "frames": int(len(sample.time_s)),
        "duration_s": float(sample.time_s[-1] - sample.time_s[0]),
        "onset_s": sample.imbalance_onset_s,
        "impact_s": sample.first_impact_s,
        "step_alarmed": step["window_alarmed"],
        "step_first_alarm_s": step["first_alarm_s"],
        "step_peak": float(np.max(step["scores"])) if len(step["scores"]) else 0.0,
        "step_latency_s": _latency(step["first_alarm_s"], sample.imbalance_onset_s),
        "dtw_alarmed": dtw["window_alarmed"],
        "dtw_first_alarm_s": dtw["first_alarm_s"],
        "dtw_peak": float(np.max(dtw["frame_scores"])) if len(dtw["frame_scores"]) else 0.0,
        "dtw_latency_s": _latency(dtw["first_alarm_s"], sample.imbalance_onset_s),
        "features": features,
        "step_scores": np.asarray(step["scores"]),
        "dtw_scores": np.asarray(dtw["frame_scores"]),
    }


def render_figure(sample: ChannelSample, view: dict, config: DashboardConfig) -> bytes:
    """Four-panel PNG (waterfall / range-time / features / detector scores)."""

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
    time = sample.time_s
    delay_ns = sample.delay_s * 1e9
    power = np.abs(sample.cir) ** 2
    waterfall = 10.0 * np.log10(power.T + 1e-30)

    axis = axes[0][0]
    image = axis.imshow(
        waterfall, aspect="auto", origin="lower", cmap="magma",
        extent=[time[0], time[-1], delay_ns[0], delay_ns[-1]],
    )
    fig.colorbar(image, ax=axis, label="|CIR| power (dB)")
    axis.set_title("CIR waterfall")

    map_ = range_time_map(
        sample.cir, sample.delay_s, time, sample.baseline_cir, config.range_time
    )
    axis = axes[0][1]
    image = axis.imshow(
        map_.magnitude.T, aspect="auto", origin="lower", cmap="magma",
        extent=[time[0], time[-1], delay_ns[0], delay_ns[-1]], vmin=0.0, vmax=1.0,
    )
    fig.colorbar(image, ax=axis, label="dB vs baseline (clipped)")
    axis.set_title(f"range-time (ref={map_.reference})")

    axis = axes[1][0]
    axis.plot(time, view["features"]["power_db"], color="tab:blue", label="power (dB)")
    axis.set_ylabel("power (dB)", color="tab:blue")
    twin = axis.twinx()
    twin.plot(time, view["features"]["rms_delay_spread_s"] * 1e9,
              color="tab:orange", label="rms delay spread (ns)")
    twin.set_ylabel("delay spread (ns)", color="tab:orange")
    axis.set_title("channel features")
    axis.set_xlabel("time (s)")

    axis = axes[1][1]
    axis.plot(time, view["step_scores"], color="tab:green", label="step detector")
    axis.plot(time, view["dtw_scores"], color="tab:purple", label="DTW template")
    axis.axhline(config.window.threshold, color="black", ls=":", lw=1, label="alarm threshold")
    axis.fill_between(
        time, 0.0, 1.05, where=view["step_scores"] >= 1.0,
        color="tab:green", alpha=0.15,
    )
    axis.set_ylim(-0.02, 1.05)
    axis.set_title("detector frame scores")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("score")
    for axis in (axes[0][0], axes[0][1], axes[1][0], axes[1][1]):
        if view["onset_s"] is not None:
            axis.axvline(view["onset_s"], color="cyan", ls="--", lw=1)
        if view["impact_s"] is not None:
            axis.axvline(view["impact_s"], color="red", ls="--", lw=1)
    handles, labels = axes[1][1].get_legend_handles_labels()
    axis.legend(handles, labels + ["onset (cyan) / impact (red)"],
                loc="upper left", fontsize=7)
    fig.suptitle(
        f"{view['sample_id']}  [{view['activity']}/{view['event_label']}]  "
        f"{view['frames']} frames / {view['duration_s']:.2f} s",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return buffer.getvalue()


def load_report_scalars(path: Path) -> dict | None:
    """The few aggregate numbers the dashboard header shows, or None."""

    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "fall_detection_rate": payload.get("fall_detection_rate"),
        "detection_latency_s_median": payload.get("detection_latency_s_median"),
        "adl_false_alarm_per_hour": payload.get("adl_false_alarm_per_hour"),
        "detector_kind": (payload.get("detector") or {}).get("kind", "?"),
    }


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def build_html(
    views: list[dict],
    figures: list[bytes],
    reports: dict[str, dict | None],
    batch_dir: Path,
    config: DashboardConfig,
) -> str:
    """Assemble the self-contained HTML page."""

    rows = []
    sections = []
    for view, figure in zip(views, figures, strict=True):
        anchor = html.escape(view["sample_id"], quote=True)
        rows.append(
            f"<tr><td><a href='#{anchor}'>{html.escape(view['sample_id'])}</a></td>"
            f"<td>{html.escape(view['activity'])}</td>"
            f"<td>{html.escape(view['event_label'])}</td>"
            f"<td>{view['frames']}</td><td>{view['duration_s']:.2f}</td>"
            f"<td>{'是' if view['step_alarmed'] else '否'}</td>"
            f"<td>{_fmt(view['step_latency_s'])}</td>"
            f"<td>{'是' if view['dtw_alarmed'] else '否'}</td>"
            f"<td>{_fmt(view['dtw_peak'])}</td></tr>"
        )
        encoded = base64.b64encode(figure).decode("ascii")
        step = _fmt(view["step_first_alarm_s"], " s")
        dtw = _fmt(view["dtw_first_alarm_s"], " s")
        sections.append(
            f"<section id='{anchor}'><h2>{html.escape(view['sample_id'])} "
            f"<small>[{html.escape(view['activity'])}/{html.escape(view['event_label'])}]</small></h2>"
            f"<p class='numbers'>台阶检测: 报警 {'是' if view['step_alarmed'] else '否'}"
            f" / 首次 {step} / 峰值 {_fmt(view['step_peak'])}"
            f" &nbsp;|&nbsp; DTW 模板: 报警 {'是' if view['dtw_alarmed'] else '否'}"
            f" / 首次 {dtw} / 峰值 {_fmt(view['dtw_peak'])}</p>"
            f"<img src='data:image/png;base64,{encoded}' alt='{anchor}'/></section>"
        )

    def report_cell(kind: str, report: dict | None) -> str:
        if report is None:
            return (f"<div class='card'><h3>{kind}</h3><p>报告未生成"
                    f"（先运行对应 evaluate 脚本）</p></div>")
        return (
            f"<div class='card'><h3>{html.escape(report['detector_kind'])}</h3>"
            f"<p>fall 检出 {_fmt(report['fall_detection_rate'])} &nbsp;"
            f"延迟中位 {_fmt(report['detection_latency_s_median'], ' s')} &nbsp;"
            f"ADL 误报 {_fmt(report['adl_false_alarm_per_hour'], '/h')}</p></div>"
        )

    cards = "".join([
        report_cell("台阶检测报告", reports.get("step")),
        report_cell("DTW 模板报告", reports.get("dtw")),
    ])
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>Sionna CIR 样本浏览器 — {html.escape(batch_dir.name)}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 2rem;
       background: #14161a; color: #d8dade; }}
a {{ color: #7ab8ff; }}
h1 {{ font-size: 1.3rem; }} h2 {{ font-size: 1.05rem; margin-top: 2rem; }}
small {{ color: #8a9099; font-weight: normal; }}
.cards {{ display: flex; gap: 1rem; margin: 1rem 0; }}
.card {{ border: 1px solid #2c313a; border-radius: 8px; padding: 0.6rem 1rem;
         background: #1b1e24; }}
.card h3 {{ margin: 0 0 0.3rem; font-size: 0.9rem; color: #a9b2bd; }}
.card p {{ margin: 0; font-size: 0.9rem; }}
table {{ border-collapse: collapse; font-size: 0.85rem; margin: 1rem 0; }}
th, td {{ border: 1px solid #2c313a; padding: 0.25rem 0.6rem; text-align: left; }}
th {{ background: #1b1e24; }}
img {{ max-width: 100%; border: 1px solid #2c313a; border-radius: 6px; }}
.numbers {{ font-size: 0.85rem; color: #a9b2bd; }}
footer {{ margin-top: 2.5rem; font-size: 0.8rem; color: #8a9099;
          border-top: 1px solid #2c313a; padding-top: 0.8rem; }}
</style></head><body>
<h1>Sionna CIR 样本浏览器 <small>{html.escape(str(batch_dir))}</small></h1>
<div class="cards">{cards}</div>
<table><tr><th>样本</th><th>类别</th><th>事件</th><th>帧数</th><th>时长(s)</th>
<th>台阶报警</th><th>台阶延迟(s)</th><th>DTW报警</th><th>DTW峰值</th></tr>
{''.join(rows)}</table>
{''.join(sections)}
<footer>
口径：这是记录批量的离线浏览器，不构成检测性能结论。图上青色虚线=标签失衡起点，
红色虚线=首次撞击；阈值 window={config.window.window_length_s}s/
stride={config.window.stride_s}s/thr={config.window.threshold}。生成于 {stamp}。
</footer></body></html>"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sionna-dir", type=Path, required=True,
                        help="directory with *.cir.npz / *.import.json pairs")
    parser.add_argument("--out", type=Path, default=None,
                        help="output HTML (default: <sionna-dir>/detection/dashboard.html)")
    parser.add_argument("--keep-pngs", action="store_true",
                        help="also write each sample's panel PNG next to the HTML")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or (args.sionna_dir / "detection" / "dashboard.html")
    config = default_config()
    samples = iter_channel_samples(args.sionna_dir)
    if not samples:
        print(f"no admitted channel samples under {args.sionna_dir}")
        return 1
    falls = [sample for sample in samples if sample.activity == "fall"]
    try:
        template = build_fall_template(
            [sample.cir for sample in falls],
            [sample.delay_s for sample in falls],
            [sample.time_s for sample in falls],
            [sample.imbalance_onset_s for sample in falls],
            config.template,
        )
    except ValueError as error:
        print(f"DTW panel disabled: {error}")
        template = None
    views = [sample_view(sample, template, config) for sample in samples]
    figures = [
        render_figure(sample, view, config)
        for sample, view in zip(samples, views, strict=True)
    ]
    if args.keep_pngs:
        panels = out.parent / "dashboard_panels"
        panels.mkdir(parents=True, exist_ok=True)
        for sample, figure in zip(samples, figures, strict=True):
            (panels / f"{sample.sample_id}.png").write_bytes(figure)
    reports = {
        "step": load_report_scalars(out.parent / "detection_report.json"),
        "dtw": load_report_scalars(out.parent / "dtw_report.json"),
    }
    page = build_html(views, figures, reports, args.sionna_dir, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"dashboard -> {out} ({out.stat().st_size / 1e6:.1f} MB, {len(views)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
