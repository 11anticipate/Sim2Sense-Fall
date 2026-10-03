"""Fix frozen resolved_configs on a ProtoMotions cloud run (2026-10-01 incident).

Root cause of the "eval EMA alpha=200" bug: a blanket sed during eval-disable
troubles flipped evaluators/config.py's eval_action_ema_alpha default from None
to 200. train_agent freezes the materialized config into
results/<exp>/resolved_configs{,_inference}.pt at launch, and inference_agent
loads THOSE frozen files ("exact reproducibility") - so later source fixes had
no effect and every full-eval ran with alpha=200 (a high-pass amplifier that
zeroed all eval scores).

This script audits and repairs the frozen files in place (set
agent.evaluator.eval_action_ema_alpha to the upstream default None). Run AFTER
cloud_setup.sh / at any fresh instance, together with patch_disable_eval.py
and patch_eval_nan.py.
"""
import py_compile
import sys
import torch
from pathlib import Path

ROOT = Path("/root/autodl-tmp/ProtoMotions/results")
EXP = sys.argv[1] if len(sys.argv) > 1 else "smpl_production_r4"
FIX = "--fix" in sys.argv

for name in ("resolved_configs_inference.pt", "resolved_configs.pt"):
    p = ROOT / EXP / name
    if not p.exists():
        print(f"skip (missing): {p}")
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    val = d["agent"].evaluator.eval_action_ema_alpha
    print(f"{name}: eval_action_ema_alpha = {val}")
    if val not in (None, 0.8) and FIX:
        d["agent"].evaluator.eval_action_ema_alpha = None
        torch.save(d, p)
        print(f"  -> fixed to None")
