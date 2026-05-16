#!/usr/bin/env bash
# Bi GraphTransPPO hierarchical policy with r2_arch: lookup (default YAML).
# For encoder_graph_shared (weight-tied R2 GraphTransformer), use
# run_genmolrl_graphtransppo_bi_hierarchical_encoder_graph_shared.sh instead.
# Pair: run_genmolrl_graphtransppo_bi_multidiscrete.sh (same lookup R2 head,
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
