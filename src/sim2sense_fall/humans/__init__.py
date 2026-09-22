"""CPU-side human body, motion and ground-truth pipeline for Sim2Sense-Fall.

The package mirrors the scene pipeline's split between *planning* (pure Python,
runs on any CPU, unit-testable) and *authoring* (:mod:`usd_human`, the only module
that needs Isaac Sim and the USD runtime).

Layer map
---------

``skeleton``
    SMPL/SMPL-H joint topology, the rest skeleton, and validated articulation
    rest geometry. No Physics, no USD.
``assets``
    The registry of licensed model files and motion-capture sources, with
    provenance and an explicit "asset not available" failure path.
``rotations``
    Axis-angle / quaternion / matrix conversions and their validity checks.
``motion``
    Immutable motion clips, AMASS (SMPL-H) retargeting onto the SMPL body
    joints, and resampling onto the unified simulation timeline.
``config``
    Strict YAML configuration for the rig, the controller, the perturbation
    scripts and the export contract.
``rig``
    The human rig *plan* -- capsule segments, mass allocation, joint axes,
    limits and drive gains -- plus CPU forward kinematics, so the same geometry
    the USD author consumes can be tested without a simulator.
``events``
    Fall event definitions (imbalance onset, first impact, final stabilisation)
    computed from trajectories rather than from the perturbation schedule.
``export``
    The unified-timeline ground-truth record, its provenance block and the
    subject/motion-based split fields.
``usd_human``
    Isaac Sim authoring and runtime sampling for the rig plan.

Nothing here downloads models, and nothing here silently substitutes a stand-in
for a licensed asset: a missing SMPL file raises
:class:`~sim2sense_fall.humans.assets.AssetUnavailable` with the exact command a
human needs to run.
"""

from __future__ import annotations

__all__ = [
    "assets",
    "config",
    "events",
    "export",
    "motion",
    "mesh_sequence",
    "rig",
    "rotations",
    "skeleton",
]
