#!/usr/bin/env bash
# Exhaustive enumeration baseline (bi). All hyperparameters / masking /
# reward / experiment name come from the YAML config below.
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm exhausted_search \
  --config "${CONFIG:-configs/exhausted_search_bi_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
