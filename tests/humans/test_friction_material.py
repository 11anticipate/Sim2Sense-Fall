"""Friction material authoring on human colliders (requires bundled OpenUSD).

The friction audit measured the human colliders with no physics material at
all: PhysX silently fell back to 0.5/0.5 combined by average with the floor,
and the audit's mu sweep could not even be phrased. These tests pin that an
explicit material is authored once, carries the configured coefficients, and
is bound by every human collider -- and that omitting ``friction`` keeps the
old behaviour (no material prim, no bindings).
"""

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("pxr.Usd", reason="requires the bundled OpenUSD runtime")

from sim2sense_fall.humans import usd_human  # noqa: E402
from sim2sense_fall.humans.config import load_human_config  # noqa: E402
from sim2sense_fall.humans.rig import plan_human_rig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

MATERIAL_PATH = "/World/Human/contact_material"


def _plan():
    return plan_human_rig(load_human_config(ROOT / "configs/humans/human_smpl_multiaxis.yaml"))


def test_friction_material_authored_and_bound():
    plan = _plan()
    runtime = usd_human.pxr_modules()
    stage = runtime.Usd.Stage.CreateInMemory()
    usd_human.author_human(stage, plan, friction=(1.45, 1.30))
    material = stage.GetPrimAtPath(MATERIAL_PATH)
    assert material.IsValid()
    assert material.GetAttribute("physics:staticFriction").Get() == pytest.approx(1.45)
    assert material.GetAttribute("physics:dynamicFriction").Get() == pytest.approx(1.30)
    assert material.GetAttribute("physxMaterial:frictionCombineMode").Get() == "average"
    bound = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(runtime.UsdPhysics.CollisionAPI):
            continue
        if not prim.GetPath().pathString.startswith("/World/Human/"):
            continue
        binding = runtime.UsdShade.MaterialBindingAPI(prim)
        direct = binding.GetDirectBinding("physics")
        assert direct.GetMaterialPath().pathString == MATERIAL_PATH, prim.GetPath()
        bound += 1
    assert bound >= 20  # the shipped rig carries 22 colliders


def test_omitting_friction_keeps_legacy_stage():
    plan = _plan()
    runtime = usd_human.pxr_modules()
    stage = runtime.Usd.Stage.CreateInMemory()
    usd_human.author_human(stage, plan)
    assert not stage.GetPrimAtPath(MATERIAL_PATH).IsValid()


def test_negative_friction_is_rejected():
    plan = _plan()
    runtime = usd_human.pxr_modules()
    stage = runtime.Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="nonnegative"):
        usd_human.author_human(stage, plan, friction=(-0.1, 0.9))


def test_foot_hull_authors_local_mesh_and_preserves_inertia():
    plan = _plan()
    foot = plan.link("left_ankle")
    vertices = ((0., 0., 0.), (.1, 0., 0.), (0., .1, 0.), (0., 0., .1))
    faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    hull = replace(foot, collision_box_bounds=None, collision_mesh_vertices=vertices,
                   collision_mesh_faces=faces)
    plan = replace(plan, links=tuple(hull if link.name == foot.name else link
                                    for link in plan.links))
    runtime = usd_human.pxr_modules()
    stage = runtime.Usd.Stage.CreateInMemory()
    usd_human.author_human(stage, plan)
    prim = stage.GetPrimAtPath(foot.capsule.path)
    assert prim.GetTypeName() == "Mesh"
    assert prim.GetAttribute("physics:approximation").Get() == "convexHull"
    assert len(prim.GetAttribute("points").Get()) == 4
    assert len(prim.GetAttribute("faceVertexIndices").Get()) == 12
    assert hull.mass_kg == foot.mass_kg
    assert hull.inertia_kg_m2 == foot.inertia_kg_m2
