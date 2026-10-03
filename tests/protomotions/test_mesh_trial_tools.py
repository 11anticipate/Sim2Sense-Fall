"""Trial-mesh tooling: zero-phase filter, contact-chatter metric, renderer."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
scipy = pytest.importorskip("scipy")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name,
        Path(__file__).resolve().parents[2] / "scripts" / "protomotions" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass(slots=True) rebuild looks the module up
    spec.loader.exec_module(module)
    return module


filter_trial_mesh = _load("filter_trial_mesh")
measure_contact_chatter = _load("measure_contact_chatter")
render_trial_mesh = _load("render_trial_mesh")


class TestFilterVertices:
    def test_removes_bounce_band_keeps_motion_band(self):
        fps = 120.0
        t = np.arange(int(fps)) / fps
        slow = np.sin(2 * np.pi * 1.0 * t)
        bounce = 0.5 * np.sin(2 * np.pi * 15.0 * t)
        verts = np.zeros((len(t), 1, 3))
        verts[:, 0, 2] = slow + bounce
        out = filter_trial_mesh.filter_vertices(verts, fps, cutoff_hz=4.0)
        mid = slice(20, -20)  # ignore filtfilt edge transients
        residual_bounce = out[mid, 0, 2] - slow[mid]
        assert np.abs(residual_bounce).max() < 0.1 * np.abs(bounce).max()
        assert np.abs(out[mid, 0, 2] - slow[mid]).max() < 0.15

    def test_rejects_cutoff_above_nyquist(self):
        with pytest.raises(ValueError):
            filter_trial_mesh.filter_vertices(np.zeros((10, 1, 3)), 120.0, 90.0)


class TestChatterMetrics:
    def test_quiet_foot(self):
        t = np.arange(240) / 120.0
        verts = np.zeros((len(t), 1, 3))
        verts[:, 0, 2] = 0.002 + 0.0005 * np.sin(2 * np.pi * 0.5 * t)
        m = measure_contact_chatter.chatter_metrics(verts, t)
        assert m["penetration_cm"] == pytest.approx(0.15, abs=0.05)
        assert m["window_peak_to_peak_cm"] < 0.3
        assert m["bounce_hz"] < 2.0

    def test_bouncing_foot(self):
        t = np.arange(480) / 120.0
        verts = np.zeros((len(t), 1, 3))
        verts[:, 0, 2] = -0.03 + 0.02 * np.sin(2 * np.pi * 6.0 * t)
        m = measure_contact_chatter.chatter_metrics(verts, t)
        assert m["penetration_cm"] == pytest.approx(-5.0, abs=1.0)
        assert m["window_peak_to_peak_cm"] == pytest.approx(4.0, abs=0.8)
        assert m["bounce_hz"] == pytest.approx(12.0, abs=2.0)


class TestRenderer:
    def _box_clip(self, name="r4"):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        frames = verts[None, :, :].repeat(5, axis=0)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4],
        ])
        return render_trial_mesh.Clip(
            name=name, frames=frames, faces=faces,
            color=(120, 155, 205), v_center=0.5, v_span=2.0, floor_z=0.0,
        )

    def test_panel_renders_nonblank(self):
        img = render_trial_mesh.render_panel(self._box_clip(), 0, 0.5, "r4 t=0.00s")
        arr = np.asarray(img)
        assert img.size == (render_trial_mesh.PANEL_W, render_trial_mesh.PANEL_H)
        colored = np.abs(arr.astype(int) - 255).sum(axis=2) > 30
        assert colored.mean() > 0.01  # silhouette present

    def test_rotate_to_travel_aligns_motion_axis(self):
        t = np.arange(100)
        verts = np.zeros((100, 1, 3))
        verts[:, 0, 0] = t * 0.01
        verts[:, 0, 1] = t * 0.01  # 45-degree diagonal travel
        rotated = render_trial_mesh.rotate_to_travel(verts)
        delta = rotated[-1, 0] - rotated[0, 0]
        assert delta[0] == pytest.approx(np.hypot(0.99, 0.99), abs=1e-6)
        assert abs(delta[1]) < 1e-6

    def test_stationary_clip_keeps_world_axes(self):
        verts = np.zeros((50, 1, 3))
        verts[:, 0, 0] = np.linspace(0, 0.2, 50)  # 0.2 m drift: below threshold
        rotated = render_trial_mesh.rotate_to_travel(verts)
        np.testing.assert_allclose(rotated, verts)
