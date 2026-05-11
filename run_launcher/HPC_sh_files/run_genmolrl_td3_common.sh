#!/usr/bin/env bash
# Shared TD3 launcher. All hyperparameters / masking / reward /
# experiment name / max_episode_len come from the YAML config; this script
# only forwards CLI overrides for env vars the user has explicitly set.
#
# Usage (from a thin wrapper):
#   exec bash "${ROOT}/run_genmolrl_td3_common.sh" \
#     "configs/<td3_variant>.yaml"   # required; default config path
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

DEFAULT_CONFIG="${1:?missing default CONFIG path}"
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

if [[ "${STAGE_DATA:-false}" == "true" ]]; then
  STAGE_ARGS=()
  if [[ -n "${STAGE_SOURCE_DIR:-}" ]]; then
    STAGE_ARGS+=(--source-dir "${STAGE_SOURCE_DIR}")
  fi
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data "${STAGE_ARGS[@]}" >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm td3 \
  --config "${CONFIG}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
