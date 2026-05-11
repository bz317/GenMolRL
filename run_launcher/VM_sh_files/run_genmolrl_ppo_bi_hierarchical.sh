#!/usr/bin/env bash
# Bi PPO via the hand-rolled `BiPPO` trainer (--algorithm ppo_bi). Uses a
# hierarchical autoregressive policy π(T) · π(R2 | s, T). The R2 head reads
# a learned per-template embedding, and the R2 mask is per-(state, T).
# All hyperparameters come from the YAML config below.
# Pair: run_genmolrl_ppo_bi_multidiscrete.sh (SB3 MaskablePPO MultiDiscrete).
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm ppo_bi \
  --config "${CONFIG:-configs/ppo_bi_hierarchical_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
