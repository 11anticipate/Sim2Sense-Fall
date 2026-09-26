"""Dashboard builder: per-sample verdicts and the assembled HTML contract."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

_SPEC = importlib.util.spec_from_file_location(
    "build_dashboard",
    Path(__file__).resolve().parents[1] / "scripts" / "sionna" / "build_dashboard.py",
)
dashboard = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dashboard  # dataclass(slots=True) rebuild looks the module up
_SPEC.loader.exec_module(dashboard)

from sim2sense_fall.detection_data import ChannelSample  # noqa: E402

TAPS = 201


def _sample(name, activity, event_label, collapse_frame=None, frames=60):
    time = np.arange(frames) * 0.05
    cir = np.zeros((frames, TAPS), dtype=np.complex128)
    cir[:, 20] = 1.0
    if collapse_frame is not None:
        cir[collapse_frame:, 20] = 10.0
    return ChannelSample(
        sample_id=name, cir_path=None, import_path=None, cir=cir,
        delay_s=np.arange(TAPS) * 1e-8, time_s=time, baseline_cir=cir[0] * 0.5,
        activity=activity,
        event_label=event_label,
        imbalance_onset_s=collapse_frame * 0.05 if collapse_frame else None,
        first_impact_s=None,
    )


CONFIG = dashboard.default_config()


def _collapse_template():
    falls = [
        _sample(f"f{i}", "fall", "push", collapse_frame=start)
        for i, start in enumerate((10, 20), start=1)
    ]
    return dashboard.build_fall_template(
        [sample.cir for sample in falls],
        [sample.delay_s for sample in falls],
        [sample.time_s for sample in falls],
        [sample.imbalance_onset_s for sample in falls],
        CONFIG.template,
    )


def test_sample_view_reports_both_detectors():
    template = _collapse_template()
    collapse = dashboard.sample_view(
        _sample("hit", "fall", "push", collapse_frame=25), template, CONFIG
    )
    flat = dashboard.sample_view(
        _sample("miss", "adl", "stand"), template, CONFIG
    )
    assert collapse["step_alarmed"] is True
    assert collapse["step_latency_s"] == pytest.approx(
        collapse["step_first_alarm_s"] - 1.25
    )
    assert collapse["dtw_peak"] > flat["dtw_peak"]
    assert flat["step_alarmed"] is False and flat["dtw_alarmed"] is False


def test_html_embeds_every_sample_and_honest_footer(tmp_path):
    template = _collapse_template()
    samples = [
        _sample("hit", "fall", "push", collapse_frame=25),
        _sample("miss", "adl", "stand"),
    ]
    views = [dashboard.sample_view(sample, template, CONFIG) for sample in samples]
    figures = [
        dashboard.render_figure(sample, view, CONFIG)
        for sample, view in zip(samples, views, strict=True)
    ]
    reports = {
        "step": {"detector_kind": "trailing_baseline_step_change",
                 "fall_detection_rate": 1.0, "detection_latency_s_median": 0.1,
                 "adl_false_alarm_per_hour": 0.0},
        "dtw": None,
    }
    page = dashboard.build_html(views, figures, reports, tmp_path, CONFIG)
    assert page.count("data:image/png;base64,") == 2
    assert "hit" in page and "miss" in page
    assert "DTW 模板报告" in page and "报告未生成" in page
    assert "不构成检测性能结论" in page
    assert "<section id='hit'>" in page


def test_report_scalars_read_or_absent(tmp_path):
    path = tmp_path / "r.json"
    assert dashboard.load_report_scalars(path) is None
    path.write_text(json.dumps({
        "detector": {"kind": "dtw_template_matching"},
        "fall_detection_rate": 0.0,
        "detection_latency_s_median": None,
        "adl_false_alarm_per_hour": 0.0,
    }), encoding="utf-8")
    scalars = dashboard.load_report_scalars(path)
    assert scalars["detector_kind"] == "dtw_template_matching"
    assert scalars["fall_detection_rate"] == 0.0
