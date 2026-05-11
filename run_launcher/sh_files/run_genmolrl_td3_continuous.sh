#!/usr/bin/env bash
# Uni TD3 with PGFS-style continuous R2 head.
# All hyperparameters / experiment name come from the YAML config below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "${ROOT}/run_genmolrl_td3_common.sh" \
  "configs/td3_uni_continuous_masked_delta_qed.yaml"
