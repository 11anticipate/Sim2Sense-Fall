"""NaN guard v2 for ProtoMotions mimic evaluator (cloud box).

v1 caught the AssertionError but the poisoned env survived: every state
setter reads back ALL envs and asserts, so nothing could cleanse it and the
next eval batch died. v2 scrubs NaN in place from the newton sim's internal
state buffers (state_0/state_1, zero-copy torch views) so all subsequent
reads/writes are finite. Idempotent; backs up; verifies with py_compile.
"""
import py_compile
import shutil
import sys
from pathlib import Path

P = Path("/root/autodl-tmp/ProtoMotions/protomotions/agents/evaluators/mimic_evaluator.py")
src = P.read_text()

if "NaN cleanse wrote" in src:
    print("ALREADY-PATCHED-V2")
    sys.exit(0)

OLD = """            try:
                obs, rewards, dones, terminated, extras = self.env.step(actions)
            except AssertionError as e:
                # A single env blowing up to NaN must not kill the whole run.
                # Eval is no-grad so weights stay clean; treat this episode as
                # failed, clear the batch envs, and continue with the next batch.
                logging.warning("Eval episode aborted by non-finite sim state: %s", e)
                try:
                    self.env.reset(env_ids, **self._get_reset_kwargs())
                except Exception as reset_err:
                    logging.warning("Eval batch reset after NaN failed: %s", reset_err)
                break
"""

NEW = """            try:
                obs, rewards, dones, terminated, extras = self.env.step(actions)
            except AssertionError as e:
                # A single env blowing up to NaN must not kill the whole run.
                # Eval is no-grad so weights stay clean; treat this episode as
                # failed. All state setters read back ALL envs and assert, so
                # the NaN must be scrubbed at the source: in-place nan_to_num
                # on the newton sim's internal state buffers (zero-copy torch
                # views), then move on to the next eval batch.
                logging.warning("Eval episode aborted by non-finite sim state: %s", e)
                try:
                    import warp as wp

                    sim = self.env.simulator
                    n = 0
                    for st in (sim.state_0, sim.state_1):
                        for m in (
                            "get_root_transforms",
                            "get_root_velocities",
                            "get_link_transforms",
                            "get_link_velocities",
                            "get_dof_positions",
                            "get_dof_velocities",
                        ):
                            fn = getattr(sim.robot_view, m, None)
                            if fn is None:
                                continue
                            t = wp.to_torch(fn(st))
                            torch.nan_to_num_(t)
                            n += 1
                    logging.warning("NaN cleanse wrote %d buffers", n)
                except Exception as cleanse_err:
                    logging.warning("NaN cleanse failed: %s", cleanse_err)
                break
"""

count = src.count(OLD)
assert count == 1, f"expected exactly 1 occurrence of v1 guard block, found {count}"

if not Path(str(P) + ".bak_nan_guard").exists():
    shutil.copy2(P, str(P) + ".bak_nan_guard")
P.write_text(src.replace(OLD, NEW))
py_compile.compile(str(P), doraise=True)
print("PATCHED-V2-OK")
