#!/usr/bin/env bash
# Bi-TD3 with the PGFS-faithful continuous R2 head.
# Hyperparameters / experiment name come from the YAML config below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "${ROOT}/run_genmolrl_td3_common.sh" \
  "configs/td3_bi_continuous_masked_delta_qed.yaml"
