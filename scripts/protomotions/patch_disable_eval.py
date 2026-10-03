"""Disable in-training evaluation for ProtoMotions (cloud box, user decision).

Every crash tonight originated in the eval phase (OOM spike, sampler-reweight
TypeError, repeated NaN blow-ups); the training loop itself never crashed.
This forces should_evaluate=False in agent.fit() so no eval ever runs during
training (scheduled and checkpoint-load evals both killed). Post-training
acceptance still uses inference_agent --full-eval, which bypasses fit().
Idempotent; backs up agent.py; verifies with py_compile.
"""
import py_compile
import shutil
import sys
from pathlib import Path

P = Path("/root/autodl-tmp/ProtoMotions/protomotions/agents/base_agent/agent.py")
src = P.read_text()

if "Eval disabled (2026-10-01" in src:
    print("ALREADY-PATCHED")
    sys.exit(0)

OLD = """            checkpoint_eval_due = self.just_loaded_checkpoint_should_evaluate
            should_evaluate = self.evaluator is not None and (
                scheduled_eval_due or checkpoint_eval_due
            )
            if should_evaluate:
"""

NEW = """            checkpoint_eval_due = self.just_loaded_checkpoint_should_evaluate
            should_evaluate = self.evaluator is not None and (
                scheduled_eval_due or checkpoint_eval_due
            )
            # Eval disabled (2026-10-01, user decision): every crash tonight
            # originated in the eval phase (OOM spike at 4096 envs, sampler
            # reweight TypeError, repeated NaN blow-ups); the training loop
            # itself has never crashed. Final acceptance happens post-training
            # via inference_agent --full-eval, which does not run through fit().
            should_evaluate = False
            if should_evaluate:
"""

count = src.count(OLD)
assert count == 1, f"expected exactly 1 occurrence of eval gate, found {count}"

shutil.copy2(P, str(P) + ".bak_disable_eval")
P.write_text(src.replace(OLD, NEW))
py_compile.compile(str(P), doraise=True)
print("PATCHED-OK")
