#!/usr/bin/env bash
# Bi GraphTransPPO hierarchical policy with R2 encoder_graph_shared
# (weight-tied Siamese: R2 keys from the same GraphTransformer as R1).
# NOT lookup — see run_genmolrl_graphtransppo_bi_hierarchical.sh for
# r2_arch: lookup (gr7aa7z6-style, stable on A100).
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm graphtransppo_bi \
  --config "${CONFIG:-configs/graphtransppo_bi_hierarchical_encoder_graph_shared_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
