#!/usr/bin/env bash
# Uni A2C. All hyperparameters / masking / reward / experiment name come from
# the YAML config below. Export env vars (REACTION_MODE, MASKING, REWARD,
# EXPERIMENT_NAME, MAX_EPISODE_LEN, ...) only when you want to override the
# YAML at launch time. See `_launcher_common.sh` for the full list.
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm a2c \
  --config "${CONFIG:-configs/a2c_uni_masked_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
