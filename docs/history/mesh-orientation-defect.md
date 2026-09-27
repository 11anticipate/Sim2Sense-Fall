# SMPL rest-pose orientation defect — root cause and repair

> Historical model-frame repair evidence. The later AMASS world/local-frame correction is recorded in
> [the 2026-09-24 audit](amass-physics-audit-2026-09-24.md). See [current human usage](human-simulation.md)
> and [documentation index](README.md) for present status; do not apply a model basis to world translations.

Status: **root-caused and fixed** (2026-09-23)
Scope: the SMPL body model loader and everything downstream (rig, skinning, mesh
export for the Sionna RT hand-off). No unrelated modules touched.

## Symptom

The standing rest body loaded by `load_smpl_model` measured **1.83 m along X and
1.80 m along Z** — a body of human size in what looked like the wrong pose — while the
rig's forward kinematics simultaneously reported correct, Z-up link positions
(`pelvis z = 1.0126`, `head z = 1.615`, `left_foot z = 0.05`).

That split is why it stayed hidden: the *links* are right, the *mesh* looks wrong, and
every check that inspects link positions passes.

## Root cause

One real defect, plus one false alarm that is recorded here because it cost more time
than the defect did.

### The real defect: `load_smpl_model` applied a yaw it could not express

The loader derived only the up axis and passed it through `up_axis_conversion`. That
function only guarantees **up stays up**; it never verifies the two horizontal axes.

`RestSkeleton` documents the pipeline body frame
(`src/sim2sense_fall/humans/skeleton.py:352-355`):

> The frame is right-handed with `+X` forward, `+Y` left and `+Z` up.

The released SMPL template uses a **different horizontal convention**. Measured
directly from the raw pickle, with no rotation applied:

| anatomical quantity | raw file vector | unit | lies along |
| --- | --- | --- | --- |
| shoulder span (lateral) | `[0.3476, 0.0008, 0.0048]` | `[0.9999, 0.0024, 0.0138]` | **X** |
| pelvis→head (up) | `[0.0068, 0.5759, 0.0083]` | `[0.0118, 0.9998, 0.0144]` | **Y** |
| hip→knee (down) | `[0.0343, -0.3752, -0.0045]` | `[0.091, -0.9958, -0.0119]` | **−Y** |
| ankle→foot (forward) | `[0.0264, -0.0558, 0.1193]` | `[0.1963, -0.4154, 0.8882]` | **+Z** |

So the file frame is

```
X = lateral (left ↔ right)      Y = up      Z = forward
```

and the pipeline frame is

```
X = forward      Y = left      Z = up
```

A correct change of basis therefore has to rotate file **X (lateral)** onto pipeline
**Y** and file **Z (forward)** onto pipeline **X** — a 90° rotation about the up axis.

`source_up` was derived *correctly* (**the file's up axis really is Y**), so
`_dominant_axis` was never at fault and its 0.99 axis-alignment gate was satisfied
legitimately. An earlier draft of this document blamed `_dominant_axis`; measurement
falsified that.

The defect was in leaning on `up_axis_conversion`
(`src/sim2sense_fall/humans/motion.py`). Its docstring states the premise:

> Both frames are right-handed and share the same forward direction (`+X`).

That premise is **false for this file**: the template's `+X` is lateral, not forward.
The function only has entries for `("y","z")` and `("z","y")`, so it cannot express a
90° yaw, and it returns

```
B = [[1, 0, 0],
     [0, 0, -1],
     [0, 1, 0]]        det = +1, proper rotation

file X -> pipeline X      <-- WRONG: lateral treated as forward
file Y -> pipeline Z      <-- right: up stays up
file Z -> pipeline -Y     <-- wrong consequence
```

Applying `B` to the anatomical vectors confirms the misalignment — the shoulder span and
the arm reach land on pipeline **X**, which the pipeline reserves for forward:

| after `B` | vector | unit | pipeline axis |
| --- | --- | --- | --- |
| pelvis→head | `[0.0068, -0.0083, 0.5759]` | `[0.0118, -0.0144, 0.9998]` | +Z up ✓ |
| left_hip→right_hip | `[-0.1372, -0.0025, 0.0009]` | `[-0.9998, -0.0182, 0.0064]` | **X** ✗ |
| left_shoulder→right_shoulder | `[-0.3476, 0.0048, -0.0008]` | `[-0.9999, 0.0138, -0.0024]` | **X** ✗ |
| left_shoulder→left_hand | `[0.5929, 0.0436, -0.0119]` | `[0.9971, 0.0733, -0.0201]` | **X** ✗ |

"Up is up" held, so the loader's own consistency check passed. The horizontal
convention was rotated 90°, and nothing noticed.

### The repair

`up_axis_conversion` is the wrong tool: a single up-axis pair cannot describe a change
of basis between two frames whose *horizontal* conventions differ. The shipped fix:

1. `body_frame_conversion(up=, forward=, left=, source=)` in
   `src/sim2sense_fall/humans/motion.py` takes a full anatomical axis triple, builds the
   rotation carrying source coordinates into the pipeline frame, and **rejects
   left-handed or repeated axis triples** (`det = +1` required), so a mirrored body
   cannot be produced silently.
2. `_derive_body_frame` in `src/sim2sense_fall/humans/assets.py` derives up from
   pelvis→head cross-checked against hip→knee, left from the hip line cross-checked
   against the shoulder line, and forward fixed by right-handedness then verified against
   the toe direction. Each axis is gated to 0.99 alignment and the evidence is recorded
   on `SmplModel.source_frame`.
3. `_check_standing_frame` refuses a body whose head is not clearly above the pelvis and
   whose feet are not clearly below it.
4. `_dominant_axis` was deleted (dead after the fix). `up_axis_conversion` itself
   **stays** — it is correct for what it claims and is still used by the AMASS retarget
   path where only the up axis matters.

After this fix the loader reports `source_frame = up=y forward=z left=x`, and the
template extent moved from `[1.5341, 0.1546, 1.4963]` to `[0.2905, 1.7451, 1.7174]`.

## The false alarm: an extent is not a pose

While confirming the fix I compared the exported first-frame extent against the
pre-fix one, saw `[0.3038, 1.8252, 1.7963]`, and read it as "a body lying down". That
reading was wrong, and the mistake is worth recording because it is the same class of
error as the defect itself — trusting a proxy instead of asking the anatomical question.

`[0.3038, 1.8252, 1.7963]` is **thickness × arm span × height** for a standing SMPL
template, whose rest pose is a T-pose with the arms out: the arm span (1.825 m)
legitimately exceeds the height (1.796 m). The template extent before skinning,
`[0.2905, 1.7451, 1.7174]`, has the same property — 1.7451 > 1.7174 — so "the arms span
more than the body is tall" is true of the correct body as well. It is not a defect
signature, and no arrangement of the three extents can distinguish a standing T-pose
from a body lying with its arms out.

What settles a pose is anatomy, and only two questions do it:

1. **Where are the head and the feet?** Skinning at the rest pose puts the vertex nearest
   the head joint at `z = +1.5955` and the vertex nearest an ankle at `z = +0.0907` —
   head well above feet, so standing.
2. **What happens across the clip?** For `fall_forward_reference`, frame 0 spans
   `z ∈ [0.031, 1.827]` with the head above the feet; by the last frame it is
   `z ∈ [0.466, 0.770]`. The body has gone from upright to flat, which is what a fall is.

The one thing the extent *did* reveal, once it was compared against the link positions
rather than against the previous artifact, is that the two can disagree — which is the
basis of the export check described below. Chasing the phantom also turned up a genuine
latent hazard in `fit_mesh_to_rest_joints`, closed here.

So the "defect 2" narrative in an earlier version of this document is retracted:
`fit_mesh_to_rest_joints` was applying the right kind of transform. The extent I had
mistaken for evidence of a lying body was the correct standing T-pose.

### What *was* worth fixing there

While chasing the phantom, a genuine latent hazard was found and closed. The function
computed its scale from a bare positional `argmin` between `mesh.rest_joints` and the
rig's `rest_joint_positions`. The rig rescales joints to the configured standing height
while the licensed file keeps its native stature, so a *correct* pairing still leaves the
arrays tens of millimetres apart — 53 mm at the ankle, 91 mm at the foot. Those gaps are
large enough that from about 1.75 m upward the positional search starts picking the wrong
row:

| configured height | positional pairing injective? | scale it would return | true uniform scale |
| --- | --- | --- | --- |
| 1.60 m | yes | 0.9791 | 0.9791 |
| 1.70 m (shipped) | yes | 1.0459 | 1.0459 |
| 1.75 m | **no** | **1.0794** | 1.0459 |
| 1.80 m | **no** | **1.1128** | not fitted |

At a `static` configuration this would have silently rescaled the skin away from the
capsules with nothing raising. `joint_row_alignment` now takes the correspondence from
the shared joint names — both arrays are indexed by the same `SkeletonTopology`, so
identity *is* the name-matched pairing — and then *confirms* it against geometry using
inter-joint distances, which a uniform scale changes by one common factor:

* the row being tested and the joint being matched are masked out of each comparison,
  because both self-distances are structurally zero and would otherwise dominate the mean;
* each rig joint's own counterpart must reproduce the rig's geometry better than any
  other mesh row, and the map must be injective.

Absolute positions cannot serve as the confirmation, and neither can bone *directions*.
Scaling about the origin moves the root itself, so a pure 1.08 scale legitimately puts
the spine joints nearer to the mesh's pelvis than to their own counterparts; and a spine
is collinear, so a direction test cannot tell `spine1` from `spine2`. Pairwise distances
are invariant under the translation, scale by a known constant, and separate collinear
joints — which is why they are the check that works.

`fit_mesh_to_rest_joints` additionally asserts the residual is a pure similarity, which
is what "uniform body scale plus local translation" means. Measured on the shipped
neutral file the residual is **0 µm**, so the assertion is free and would fire the moment
a frame mismatch is introduced.

## Why the earlier acceptance tests could not catch the real defect

| check | why it passes on a broken body |
| --- | --- |
| `0.8 <= stature <= 2.5` | a yawed body is still human-sized |
| FK vs independent CPU chain, 24 links < 5 mm | rotation-invariant; compares FK to FK |
| height fit converged to `1e-6` | uniform scale commutes with rotation |
| capsule↔joint containment | both rotated together |
| "mesh moves during the trial" positive control | motion is preserved under rotation |
| extent-based "is it standing?" heuristic | a T-pose arm span *is* taller than the body |

Every one of these is invariant under a yaw about the vertical, which is exactly what the
defect produced. The lesson that generalises: **a rotation-invariant check cannot detect a
rotation, and an extent is not a pose** — at least one acceptance test has to name an
anatomical direction explicitly.

Two yaw-visible checks are now enforced at the export boundary, and both compare the
**mesh against the links** rather than links against links — a links-only check passes on
the broken export, which is precisely how the defect survived:

| check | what it pins | shipped result |
| --- | --- | --- |
| head above feet in the standing first frame | pitch | head surface `z = +1.5955`, ankles `z = +0.0907` |
| mesh and links agree on the wide horizontal axis | yaw (facing) | both `y`: 1.8252 m against 0.3038 m |

The second is the one that catches this defect. A **positive control** confirms it is
load-bearing: yawing a corrective copy of the exported mesh by 90° about Z — reproducing
the pre-fix extent `[1.8252, 0.3038, 1.7963]` exactly — makes both checks fail, with the
message naming `mesh lateral axis 'x'` against `links lateral axis 'y'`. A check that
cannot be made to fail is not evidence, so this control is the point of the exercise.

## Verification performed

Each item was measured, not inferred.

1. Raw-file vertical identified as **Y** from `pelvis→head`, with hip→knee/ankle
   descending along −Y. Measured from the unpickled arrays.
2. Source frame triple established: lateral = X, up = Y, forward = Z. Measured.
3. Loader reports `source_frame = up=y forward=z left=x`. Observed.
4. Imported `pelvis→head` ≈ `+Z`; cosines against the procedural skeleton ≥ 0.9965.
   Measured.
5. Shoulder span and arm reach lie on **Y**, not X. Measured.
6. Proper rotation, `det = +1` (no mirroring) — enforced by `body_frame_conversion` and
   covered by a test that feeds it a left-handed triple.
7. Standing rest mesh extent `[0.2905, 1.7451, 1.7174]`: X is the thin axis. Measured.
8. **Head above ankles in world Z** at the rest pose. Asserted as a test.
9. The trajectory is correct independently of the skin: `head-pelvis` Z goes from
   `+0.6024` to `-0.0087` over `fall_forward_reference`, and the root drops `0.40 m` —
   as expected, since the trajectory comes from FK, not from the skin.
10. **The exported collection passes two independent verifiers.** The collector's own
    `--verify-only` and a separate `scripts/humans/verify_fall_collection.py` that imports
    nothing from the collector and recomputes every value from the `.npz` files with its
    own arithmetic. Both report all checks passing on all 4 fall samples. Agreement between
    two independent implementations is evidence; a suite agreeing with itself is not.
11. **The yaw control fails.** A corrective copy of the export with the mesh rotated 90°
    about Z is rejected by both verifiers, with the failing check named.
12. **The manifest is self-describing**: the licensed file's sha256, its stature, its
    vertex and face counts, and the measured `source_frame` are all recorded, so the
    import can be reproduced and audited from the manifest alone.
