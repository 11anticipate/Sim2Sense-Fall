"""Enabling self-collision must not flood the solver with rest-pose overlaps.

The torso is a stack of short bones with large radii, so neighbouring capsules
overlap in the rest pose by design -- and because multi-axis joints chain
through zero-length proxy links, PhysX sees those pairs as non-adjacent and
would collide them from the first frame. The shipped rig keeps self-collision
off, which gives up real limb-limb contact. The official Isaac Sim workflow
(OpenUSD tuning tutorial 3, "Collider pairs") is to enable self-collision and
author ``UsdPhysics.FilteredPairsAPI`` for every overlapping pair. These tests
pin the pair computation on the shipped plan and its USD authoring.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sim2sense_fall.humans import usd_human
from sim2sense_fall.humans.config import load_human_config
from sim2sense_fall.humans.rig import _world_separation, plan_human_rig, rest_overlap_pairs

ROOT = Path(__file__).resolve().parents[2]

MULTIAXIS = ROOT / "configs/humans/human_smpl_multiaxis.yaml"


def _plan(*, self_collisions: bool = False, filter_rest_overlap: bool = True):
    config = load_human_config(MULTIAXIS)
    config = replace(
        config,
        rig=replace(
            config.rig,
            self_collisions=self_collisions,
            self_collision_filter_rest_overlap=filter_rest_overlap,
        ),
    )
    return plan_human_rig(config)


def test_rest_overlap_pairs_match_the_planned_count():
    plan = _plan()
    pairs = rest_overlap_pairs(plan.links)
    assert len(pairs) == plan.stats["rest_capsule_overlap_pairs"] > 0
    # torso capsules chain through proxy links, so PhysX would treat these as
    # non-adjacent: they are the reason enabling self-collision naively explodes
    for expected in (("pelvis", "spine1"), ("spine1", "spine2"), ("left_hip", "right_hip")):
        assert expected in pairs


def test_rest_overlap_pairs_are_exactly_the_interpenetrating_ones():
    plan = _plan()
    links = [link for link in plan.links if link.capsule is not None]
    pairs = set(rest_overlap_pairs(links))
    for index, left in enumerate(links):
        for right in links[index + 1 :]:
            separated = _world_separation(left, right) >= -1e-6
            key = tuple(sorted((left.name, right.name)))
            assert (key in pairs) != separated, key


def test_shipped_plan_keeps_self_collisions_off_with_filter_available():
    plan = _plan()
    assert plan.self_collisions is False
    assert plan.self_collision_filter_rest_overlap is True


def test_rig_config_accepts_the_explicit_filter_key(tmp_path):
    source = MULTIAXIS.read_text(encoding="utf-8")
    target = tmp_path / "rig.yaml"
    target.write_text(
        source.replace(
            "self_collisions: false",
            "self_collisions: true\n  self_collision_filter_rest_overlap: true",
        ),
        encoding="utf-8",
    )
    config = load_human_config(target)
    assert config.rig.self_collisions is True
    assert config.rig.self_collision_filter_rest_overlap is True


def _authored(plan):
    pytest.importorskip("pxr.Usd", reason="requires the bundled OpenUSD runtime")
    runtime = usd_human.pxr_modules()
    stage = runtime.Usd.Stage.CreateInMemory()
    report = usd_human.author_human(stage, plan)
    return runtime, stage, report


def _filtered_pair_prims(runtime, stage):
    prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(runtime.UsdPhysics.FilteredPairsAPI):
            prims.append(prim)
    return prims


def test_self_collisions_on_filters_every_rest_overlap():
    plan = _plan(self_collisions=True)
    expected = rest_overlap_pairs(plan.links)
    runtime, stage, report = _authored(plan)
    root = stage.GetPrimAtPath(usd_human.HUMAN_ROOT_PATH)
    assert root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert report["self_collisions"] is True
    assert report["self_collision_filtered_pairs"] == [
        f"{name_a}|{name_b}" for name_a, name_b in expected
    ]
    prims = _filtered_pair_prims(runtime, stage)
    assert len(prims) == len(expected)
    # each pair is authored once, on the first capsule, targeting the second;
    # the USD relationship blocks the collision in both directions
    by_name = {prim.GetPath().pathString: prim for prim in prims}
    name_a, name_b = ("left_hip", "right_hip")
    prim = by_name[plan.link(name_a).capsule.path]
    targets = prim.GetRelationship("physics:filteredPairs").GetTargets()
    assert [path.pathString for path in targets] == [plan.link(name_b).capsule.path]


def test_self_collisions_off_keeps_the_legacy_stage():
    plan = _plan()
    runtime, stage, report = _authored(plan)
    root = stage.GetPrimAtPath(usd_human.HUMAN_ROOT_PATH)
    assert root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is False
    assert report["self_collisions"] is False
    assert report["self_collision_filtered_pairs"] == []
    assert _filtered_pair_prims(runtime, stage) == []


def test_filter_disabled_keeps_self_collisions_without_filtered_pairs():
    plan = _plan(self_collisions=True, filter_rest_overlap=False)
    runtime, stage, report = _authored(plan)
    root = stage.GetPrimAtPath(usd_human.HUMAN_ROOT_PATH)
    assert root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert report["self_collision_filtered_pairs"] == []
    assert _filtered_pair_prims(runtime, stage) == []
