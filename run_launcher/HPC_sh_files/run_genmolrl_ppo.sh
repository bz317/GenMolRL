#!/usr/bin/env bash
# Uni PPO (SB3 MaskablePPO Discrete). All hyperparameters / masking / reward /
# experiment name come from the YAML config below. To override anything at
# launch time, export the corresponding env var (e.g. MASKING=none) — see
# `_launcher_common.sh` for the full list. The launcher itself never overrides
# the YAML by default.
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm ppo \
  --config "${CONFIG:-configs/ppo_uni_masked_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
