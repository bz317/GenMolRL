#!/usr/bin/env bash
# Bi GraphTransPPO via the hand-rolled `GraphTransBiPPO` trainer
# (--algorithm graphtransppo_bi). Hierarchical autoregressive policy
# π(T | R1) · π(R2 | R1, T) where R1 is encoded by a GraphTransformer
# over the molecular graph instead of a Morgan fingerprint. All
# hyperparameters come from the YAML config below.
# Pair: run_genmolrl_graphtransppo_bi_multidiscrete.sh (same encoder,
# independent T / R2 axis heads).
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm graphtransppo_bi \
  --config "${CONFIG:-configs/graphtransppo_bi_hierarchical_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
