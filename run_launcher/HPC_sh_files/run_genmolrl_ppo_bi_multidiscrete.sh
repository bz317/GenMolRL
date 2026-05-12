#!/usr/bin/env bash
# Bi PPO with the independent (T, R2) policy factorisation:
#     π(T, R2 | s) = π_T(T | s) · π_R2(R2 | s)
# This used to dispatch to SB3 ``MaskablePPO`` over MultiDiscrete([T+1, R2]),
# but the hand-rolled BiPPO trainer now supports both ``policy_arch:
# hierarchical`` and ``policy_arch: multidiscrete`` AND the encoder-based
# R2 head (``r2_arch: encoder``), so the SB3 code path is no longer needed.
# Routing this launcher through ``--algorithm ppo_bi`` makes ppo_bi_
# multidiscrete pick R2 from the test pool at evaluate() time — comparable
# to ppo_bi_hierarchical, graphtransppo_bi, and the search baselines.
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
