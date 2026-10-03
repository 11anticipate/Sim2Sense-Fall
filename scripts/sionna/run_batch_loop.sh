#!/usr/bin/env bash
# Drive batch_generate.py + run_batch.sh until a batch has nothing left pending.
#
# run_batch.sh runs under `set -euo pipefail`, so one failed stage aborts that pass.
# Re-expanding resumes exactly the stages that are still missing -- which is the whole
# point of the generator being idempotent -- so a transient Isaac or GPU-scheduling
# failure costs one pass, not the batch. The loop is bounded so a genuinely broken stage
# cannot spin forever.
#
# Usage:
#   bash scripts/sionna/run_batch_loop.sh configs/sionna/batch_train02.yaml \
#       artifacts/batches/train02 12
# Any further arguments are forwarded to every batch_generate.py expansion, e.g.
#   --session-chunk-index 0 --session-chunk-count 2
# to keep half the sessions' 120 Hz mesh on disk at a time when the filesystem is tight.
set -uo pipefail

plan="${1:?usage: run_batch_loop.sh <plan.yaml> <batch-out> [max-passes] [batch_generate args...]}"
out="${2:?usage: run_batch_loop.sh <plan.yaml> <batch-out> [max-passes] [batch_generate args...]}"
max_passes="${3:-12}"
shift 2
if [[ $# -gt 0 ]]; then
  max_passes="$1"
  shift
fi
extra=("$@")

cd "$(dirname "$0")/../.." || exit 2
if [[ ! "$max_passes" =~ ^[0-9]+$ ]] || (( max_passes < 1 )); then
  echo "max-passes must be a positive integer, got '$max_passes'" >&2
  exit 2
fi
if [[ ! -f "$plan" ]]; then
  echo "no plan at $plan" >&2
  exit 2
fi

STAGES='keyboard\.py|export_session_mesh|import_fall_mesh|simulate\.py'

pending_stages() {
  local count
  count=$(grep -cE "$STAGES" "$out/run_batch.sh" 2>/dev/null || true)
  echo "${count:-0}"
}

expand() {
  PYTHONPATH=src python3 scripts/sionna/batch_generate.py --plan "$plan" --out "$out" \
    ${extra[@]+"${extra[@]}"}
}

for pass in $(seq 1 "$max_passes"); do
  if ! expand; then
    echo "pass $pass: expansion failed" >&2
    exit 1
  fi
  pending=$(pending_stages)
  echo "loop pass $pass: $pending pending stage commands"
  if (( pending == 0 )); then
    echo "batch converged: nothing pending"
    exit 0
  fi
  if ! bash "$out/run_batch.sh"; then
    echo "loop pass $pass: a stage failed; re-expanding to resume" >&2
  fi
done

# The last pass may have completed the final stage successfully; only a fresh
# expansion proves the batch is empty. Reporting "did not converge" without it made a
# finished train02 batch exit 1.
if ! expand; then
  echo "final expansion failed" >&2
  exit 1
fi
pending=$(pending_stages)
if (( pending == 0 )); then
  echo "batch converged after $max_passes passes: nothing pending"
  exit 0
fi
echo "batch did not converge after $max_passes passes; $pending stage commands still pending" >&2
exit 1
