"""Retarget/projection results are cached: identical inputs must load identical outputs.

The per-frame retarget loop (~8 s for the 830-frame get-up clip) and the posture
contact projection (~1 s each) were the largest prepare() costs after Isaac
itself. The cache is keyed by (source sha256, spec, rig fingerprint, dt); these
tests pin the contract that matters -- a cache hit must be indistinguishable
from a fresh compute, and a changed key must not collide.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sim2sense_fall.humans.actions import load_action_clip, load_posture
from sim2sense_fall.humans.assets import select_body
from sim2sense_fall.humans.contact_control import fit_collision_capsules
from sim2sense_fall.humans.rig import fit_rest_skeleton, plan_human_rig
from sim2sense_fall.humans.teleop import load_keyboard_config

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYBOARD_YAML = REPO_ROOT / "configs/humans/keyboard.yaml"
DT_S = 1.0 / 120.0
sys.path.insert(0, str(REPO_ROOT / "scripts" / "humans"))


@pytest.fixture(scope="module")
def plan_and_settings(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("clip_cache")
    previous = os.environ.get("SIM2SENSE_CLIP_CACHE_DIR")
    os.environ["SIM2SENSE_CLIP_CACHE_DIR"] = str(cache_dir)
    try:
        from common import DEFAULT_MOTIONS, load_inputs  # noqa: PLC0415

        settings = load_keyboard_config(KEYBOARD_YAML, REPO_ROOT)
        config, registry, _ = load_inputs(
            config_path=settings["rig"], assets_path=settings["assets"],
            motions_path=DEFAULT_MOTIONS
        )
        body = select_body(registry, model_id=config.skeleton.model_asset, allow_procedural=False)
        mesh = body.model.mesh()
        rest = fit_rest_skeleton(config, mesh.rest_skeleton())
        built = plan_human_rig(config, rest=rest, spawn_xy=tuple(settings["spawn_xy"]))
        built, _ = fit_collision_capsules(
            built, mesh, **{k: v for k, v in settings["collision_fit"].items() if k != "enabled"}
        )
        yield built, settings
    finally:
        if previous is None:
            os.environ.pop("SIM2SENSE_CLIP_CACHE_DIR", None)
        else:
            os.environ["SIM2SENSE_CLIP_CACHE_DIR"] = previous


def test_clip_cache_hit_is_identical_to_fresh_compute(plan_and_settings):
    plan, settings = plan_and_settings
    first = load_action_clip(settings["fall_replay"], plan, dt_s=DT_S)
    second = load_action_clip(settings["fall_replay"], plan, dt_s=DT_S)
    assert second.provenance == first.provenance, "a cache hit must keep the provenance"
    for name in ("joints", "heights_m", "offsets_xy", "tilts"):
        assert np.array_equal(getattr(second, name), getattr(first, name)), (
            f"cache hit changed {name}"
        )
    assert float(second.frame_dt_s) == float(first.frame_dt_s)


def test_changed_spec_gets_a_different_cache_entry(plan_and_settings):
    plan, settings = plan_and_settings
    spec = settings["fall_replay"]
    a = load_action_clip(spec, plan, dt_s=DT_S)
    b = load_action_clip({**spec, "start_s": float(spec["start_s"]) + 0.5}, plan, dt_s=DT_S)
    assert not np.array_equal(a.joints, b.joints), "a different crop must not hit the same entry"


def test_posture_cache_hit_is_identical_to_fresh_compute(plan_and_settings):
    plan, settings = plan_and_settings
    first = load_posture(settings["crouch"], plan)
    second = load_posture(settings["crouch"], plan)
    assert np.array_equal(second.joints, first.joints)
    assert second.height_m == pytest.approx(first.height_m)
    assert second.provenance == first.provenance
    assert second.contacts == first.contacts
