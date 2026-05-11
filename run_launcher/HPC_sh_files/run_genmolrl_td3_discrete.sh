#!/usr/bin/env bash
# Uni TD3 with template-only / discrete critic.
# All hyperparameters / experiment name come from the YAML config below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "${ROOT}/run_genmolrl_td3_common.sh" \
  "configs/td3_uni_discrete_masked_delta_qed.yaml"
