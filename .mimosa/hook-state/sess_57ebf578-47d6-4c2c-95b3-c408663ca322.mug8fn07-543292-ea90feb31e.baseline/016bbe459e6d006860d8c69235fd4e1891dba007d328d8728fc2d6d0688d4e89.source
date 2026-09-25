"""Indoor scene specifications, CPU planning and optional USD integration.

Use ``load_scene_spec`` followed by ``plan_scene`` for a CPU-only workflow.
USD authoring, inspection and verification live in the ``usd``, ``view`` and
``verification`` modules; USD/Isaac imports remain deferred until runtime use.
"""

from .planner import ScenePlan, ScenePrim, plan_scene, write_manifest
from .spec import FurnitureSpec, OpeningSpec, RoomSpec, SceneSpec, load_scene_spec

__all__ = [
    "FurnitureSpec",
    "OpeningSpec",
    "RoomSpec",
    "ScenePlan",
    "ScenePrim",
    "SceneSpec",
    "load_scene_spec",
    "plan_scene",
    "write_manifest",
]
