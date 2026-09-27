# AMASS motion could not be replayed — root cause and repair

> Dated repair evidence (2026-09-24). Later world/local-frame and contact fixes are documented in
> [the physics audit](amass-physics-audit-2026-09-24.md). Current scope: [plan](../task_plan.md);
> usage and evidence index: [documentation](README.md). A rig baseline pass is not full action acceptance.

Status: **root-caused and fixed, CPU and Isaac Sim both measured** (2026-09-24).
The 57-DOF multi-axis rig passes `scripts/humans/verify.py` end to end in Isaac Sim.
Scope: the clip-to-DOF mapper, the rig planner's multi-axis chain, the human config
validator, the human configs, the runtime's DOF-order translation and drive scaling.
No scene, sensor or template module touched.

## Symptom

Every AMASS clip was rejected. The screen reported "0 clips pass" and the simulator
raised *"the reference motion rotates a joint about an axis the rig cannot express"* on
the first frame of any recorded motion. A 120-clip sample produced 28 fall candidates
and **0** that the rig could express.

That result is indistinguishable from "this corpus has no falls in it", which is how
the whole AMASS library came to be written off as unusable.

## Root causes

Four defects, in the order they were reached. The first is the one that blocked the
work; the others were only reachable once the one before it stopped hiding them.

### 1. The mapper refused every real joint rotation

`joint_values_from_clip` projected each joint's axis-angle vector onto a **single**
declared axis and raised unless the off-axis part was below 1e-6 rad. A real SMPL joint
rotation is a full 3-DOF rotation, so nothing real could ever pass. Measured on 1872
frames from 60 clips, the single-axis miss is not marginal:

| joint | median miss | p95 miss |
|---|---|---|
| `left_collar` | 26.3° | 40.4° |
| `left_elbow` | 17.7° | 44.9° |
| `right_shoulder` | 18.1° | 43.4° |
| `left_wrist` | 15.2° | 49.9° |
| `left_knee` | 10.4° | 26.1° |

The missing piece was not data and not the frame conversion — both were already right.
It was a **clip-to-DOF decomposition**: one revolute joint has one degree of freedom, so
a joint that must reproduce an arbitrary rotation needs a *chain* of them.

### 2. Two axes are not enough either

Before writing any config, the same sample was measured against 2-axis chains. They
still miss 1.6–12° in the median (up to 36° at p95) at the shoulders, elbows, collars
and wrists, because a chain of two revolute joints spans a 2-dimensional subset of the
rotation group, not the whole group. Only **three** distinct axes span `SO(3)`, which is
why every moving joint in the shipped AMASS rig carries three axes.

### 3. The proxy chain dropped the bone offset — the body collapsed onto its root

`plan_human_rig` gave **every** joint of a multi-axis chain `local_pos0 = (0,0,0)`. The
first proxy is the link that hangs off the joint's own SMPL parent, so it is the one
link that must carry the bone offset; with it missing, forward kinematics placed the
entire body at the origin:

```
single-axis rig   pelvis [0 0 0]  left_hip [0 0.0936 -0.0468]  left_ankle [0 0.0936 -0.8995]
multi-axis rig    pelvis [0 0 0]  left_hip [0 0 0]             left_ankle [0 0 0]
```

The plan still reported 62 links, 19 colliders and the right mass total, and
`validate_plan_geometry` actively **enforced** the mistake ("a proxy joint must sit at
its parent's origin"). The only visible symptom was a standing height of **0.803 m for a
1.70 m figure** — which no link count, mass check or "the plan built" assertion notices.
The existing unit test for multi-axis joints checked that proxies exist and that mass is
conserved; it never checked that the body is still where it was.

### 4. The config validator read the wrong axis's limits

`HumanConfig.__post_init__` validated a neutral-pose value against `limits_deg[0]`, but
`apply_neutral_pose` writes it into the slot of the axis that the **named** DOF actually
carries — the last declared axis, which belongs to the real link. On a single-axis joint
those are the same number, which is why it went unnoticed; on a three-axis joint it
checked a proxy axis, rejecting valid poses and admitting invalid ones.

### 5. The residual metric could not see an exact answer

The decomposition's residual was `acos((trace(R) - 1) / 2)`, whose numerical noise floor
near zero is about **1.4e-8 rad**. Every exact decomposition therefore looked like a
1e-8 miss, which is larger than any sensible tolerance; a gate written against it would
have rejected the correct result. The `atan2` form (skew magnitude against the trace) is
accurate over the whole range.

### 6. Isaac ordered the articulation's DOFs by link traversal, not plan order

Found the first time the multi-axis rig reached the simulator. PhysX interleaves the
per-joint proxy chains breadth-first (`left_hip__dof1, right_hip__dof1, spine1__dof1,
left_hip__dof2, ...`), while the plan declares them depth-first per joint. On the
single-axis rig the two orders coincide, which is why every earlier verification passed.
`HumanRuntime` now computes a `dof_permutation` and translates at the runtime boundary
on every read and write (`set_joint_positions`, targets, limits, gains, all state
readers); the verify DOF check asserts set equality plus the permutation capability.

### 7. `set_control_scale(0)` also zeroed passive damping — the ragdoll NaN'd the solver

The multi-axis verification stayed red after #6 was fixed: the gravity block raised
`ValueError: non-finite world position for link 'pelvis'`. `scripts/humans/probe_gravity.py`
replayed verify's phases with per-step monitoring and located the divergence in the fall
window, with this signature:

* drives scaled to 0 removed the joint **damping** as well as the stiffness, so the
  57-DOF ragdoll folded ballistically: 975 deg/s of joint speed within the 0.2 s
  pre-baseline hold, a 113.9° fold carried into the lift, 2864 deg/s mid-fall;
* that state hit the floor and the PhysX solver produced NaN within 56 steps.

Passive joint damping is a property of the joint, not of the controller. `set_control_scale`
now scales the active **stiffness** only and always rewrites the authored damping; a body
that lost control is a damped ragdoll, not a frictionless one. Damping cannot track a
position target (negative control stays valid) and cannot hold a pose (gravity control
stays real). After the fix the same sequence folds 12.6° in the hold, peaks at ~73 deg/s
mid-fall and comes to rest on the floor. The verification lift also now starts from the
rest pose instead of inheriting the arbitrary fold, and the settle window was extended
from 1.5 s to 3 s: the 62-link body was still sliding at 0.23 m/s when the 1.5 s window
closed (46 mm of creep against the 5 mm gate).

## Repair

* **`rotations.split_rotation`** — decomposes a rotation into a chain of canonical-axis
  rotations. One axis keeps the historical projection semantics exactly. Two axes are
  solved in closed form via a relabelling onto `x, y`; three axes use the Euler closed
  form on the relabelled target, with both branches evaluated. An odd axis order is a
  reflection, so the solved angles are negated and the declared limits are reflected to
  match. Among equally exact solutions it returns the one that best respects the declared
  limits and is least extreme, because an angle that fits the pose but not the joint is a
  replay that is wrong in a way nothing else reports. The residual is geodesic.
* **`rig.axis_residuals` / `joint_values_from_clip`** — group the plan's DOFs by skeleton
  joint and decompose each group as one chain. Single-axis joints keep their exact old
  numbers; a group's DOFs share one residual.
* **`rig.plan_human_rig`** — the first proxy of a chain carries the bone offset; the rest
  sit on that point. `validate_plan_geometry` now permits exactly that arrangement.
* **`config.HumanConfig`** — a neutral pose is checked against the limits of the DOF that
  will receive it (`limits_deg[-1]`).
* **`configs/humans/human_smpl_multiaxis.yaml`** — every joint that recorded motion
  rotates gets three axes, generated by `scripts/humans/gen_multiaxis_config.py` so the
  orders, limits and gains have one authoritative definition.
* **`scripts/humans/measure_axes.py`** — the measurement that decided all of the above,
  and the check that keeps it honest.

## Evidence

`scripts/humans/measure_axes.py`, 40 clips / 1047 frames sampled from the local corpus
(2100 readable sequences, 98 unreadable in the source data):

| | single-axis rig | multi-axis rig |
|---|---|---|
| frames the rig can express | **0.0000 %** | **100.0000 %** |
| judged angles clipped by limits | 265 | **0** |
| driven joints | 14 | 19 |
| degrees of freedom | 14 | 57 |
| standing height | 1.7000 m | 1.7000 m (identical FK, 0.0 m link difference) |

The tool also fails if the corpus rotates a joint the rig does not drive, so the "feet
and hands do not need axes" assumption cannot rot silently.

CPU: `python -m pytest -q` → 266 passed, 9 skipped. `ruff` clean, `compileall` clean.

## Isaac Sim evidence (2026-09-24)

`~/isaacsim/python.sh scripts/humans/verify.py --config configs/humans/human_smpl_multiaxis.yaml
--out artifacts/humans/verify_multiaxis` → **`human verify: PASSED`** (exit 0):

| check | measured |
|---|---|
| runtime DOF set vs plan | 57/57, PhysX traversal order translated at the runtime boundary |
| joint limits USD deg ↔ Isaac rad | largest mismatch 0.000007° |
| physics pose vs CPU FK (62 links, 6 single-joint probes) | **0.00 mm** on every probe |
| tracking target falsifiability | smallest 18.75° > 15° tolerance |
| PD tracking, one DOF at a time | largest error **1.551°** (tolerance 15°) |
| drives-off negative control | ungoverned error 85.000° ≫ 15° |
| contact channel | readable (`physx_contact_report`), attribution=geometry, limbs `[left_ankle, right_ankle]` |
| gravity drop, unactuated | descended 1.1031 m from 1.2381 m; ends at z = 0.1350 m, at rest (creep < 5 mm) |
| floor penetration after settling | lowest body point −0.0000 m (tolerance −0.05 m) |

The "not yet observed" list from the CPU-only stage is closed: authoring, DOF mapping,
kinematic agreement, PD tracking of a multi-axis chain, drive scaling, contact attribution
and the penetration gate are all now measured on the 57-DOF rig.
