"""Per-joint maxJointVelocity authoring (requires bundled OpenUSD).

The statue-looking fall was traced to the drive damping kept during the
collapse, and releasing that damping too reopens a measured solver-NaN regime
(ballistic joint speeds of 2864 deg/s NaN the PhysX solver on floor impact).
The fix closes the regime at the joint instead: a per-joint velocity clamp
that normal actions never approach. These tests pin that the clamp is authored
on every driven revolute joint when configured, is omitted for legacy stages,
and that invalid values are refused.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr.Usd", reason="requires the bundled OpenUSD runtime")

from sim2sense_fall.humans import usd_human  # noqa: E402
from sim2sense_fall.humans.config import load_human_config  # noqa: E402
from sim2sense_fall.humans.rig import plan_human_rig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

JOINTS_ROOT = "/World/Human/Joints"


def _plan():
    return plan_human_rig(load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml"))


def _joint_prims(runtime, stage):
    return [prim for prim in stage.GetPrimAtPath(JOINTS_ROOT).GetAllChildren()]


def test_velocity_limit_authored_on_every_driven_joint():
    """The configured rad/s value is authored in USD's angular unit (deg/s).

    Authoring the rad/s number directly clamped the drives to "20 deg/s": the
    whole body crawled at p95 0.35 rad/s with 40 deg tracking errors. The
    authored attribute must carry the conversion.
    """

    plan = _plan()
    runtime = usd_human.pxr_modules()
    stage = runtime.Usd.Stage.CreateInMemory()
    report = usd_human.author_human(stage, plan, joint_velocity_limit_rad_s=20.0)
    assert report["joint_velocity_limit_rad_s"] == pytest.approx(20.0)
    joint_prims = _joint_prims(runtime, stage)
    assert len(joint_prims) == len(plan.joints)
    for prim, joint in zip(joint_prims, plan.joints, strict=True):
        assert prim.GetPath().pathString.endswith(joint.name)
        attr = prim.GetAttribute("physxJoint:maxJointVelocity")
        assert attr.IsValid(), prim.GetPath()
        assert attr.Get() == pytest.approx(np.rad2deg(20.0)), prim.GetPath()


def test_omitting_velocity_limit_keeps_legacy_stage():
    plan = _plan()
    runtime = usd_human.pxr_modules()
    stage = runtime.Usd.Stage.CreateInMemory()
    report = usd_human.author_human(stage, plan)
    assert report["joint_velocity_limit_rad_s"] is None
    for prim in _joint_prims(runtime, stage):
        attr = prim.GetAttribute("physxJoint:maxJointVelocity")
        assert not attr.IsValid() or attr.Get() is None, prim.GetPath()


def test_invalid_velocity_limit_is_rejected():
    plan = _plan()
    runtime = usd_human.pxr_modules()
    stage = runtime.Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="finite and positive"):
        usd_human.author_human(stage, plan, joint_velocity_limit_rad_s=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        usd_human.author_human(stage, plan, joint_velocity_limit_rad_s=-3.0)
