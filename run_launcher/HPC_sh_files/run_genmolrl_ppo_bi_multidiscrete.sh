#!/usr/bin/env bash
# Bi PPO via SB3 `MaskablePPO` over `MultiDiscrete([T+1, R2])`.
# π(T) and π(R2) are independent per-axis masked categoricals.
# All hyperparameters / masking / reward come from the YAML config below.
# Pair: run_genmolrl_ppo_bi_hierarchical.sh (hand-rolled hierarchical
# T → R2|T trainer via --algorithm ppo_bi).
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm ppo \
  --config "${CONFIG:-configs/ppo_bi_multidiscrete_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
