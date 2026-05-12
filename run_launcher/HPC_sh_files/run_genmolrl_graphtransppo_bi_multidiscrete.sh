#!/usr/bin/env bash
# Bi GraphTransPPO via the hand-rolled `GraphTransBiPPO` trainer
# (--algorithm graphtransppo_bi). Multidiscrete policy
# π(T | R1) · π(R2 | R1) — T and R2 are sampled independently given the
# graph-encoded trunk z(R1). All hyperparameters come from the YAML
# config below.
# Pair: run_genmolrl_graphtransppo_bi_hierarchical.sh (same encoder,
# T-conditional R2 head).
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm graphtransppo_bi \
  --config "${CONFIG:-configs/graphtransppo_bi_multidiscrete_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
