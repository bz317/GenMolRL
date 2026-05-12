#!/usr/bin/env bash
# Bi PPO with the independent (T, R2) policy factorisation:
#     π(T, R2 | s) = π_T(T | s) · π_R2(R2 | s)
# Routes through the hand-rolled ``BiPPO`` trainer (--algorithm ppo_bi) with
# ``policy_arch: multidiscrete`` + ``r2_arch: encoder``; see the matching
# HPC launcher run_launcher/HPC_sh_files/run_genmolrl_ppo_bi_multidiscrete.sh
# for the full rationale (encoder R2 generalisation, test-pool eval).
# Pair: run_genmolrl_ppo_bi_hierarchical.sh (autoregressive T → R2|T).
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm ppo_bi \
  --config "${CONFIG:-configs/ppo_bi_multidiscrete_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
