#!/usr/bin/env bash
# Random search baseline (uni). All hyperparameters / masking / reward /
# experiment name come from the YAML config below.
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if [[ "${STAGE_DATA:-false}" == "true" ]]; then
  STAGE_ARGS=()
  if [[ -n "${STAGE_SOURCE_DIR:-}" ]]; then
    STAGE_ARGS+=(--source-dir "${STAGE_SOURCE_DIR}")
  fi
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data "${STAGE_ARGS[@]}" >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm random_search \
  --config "${CONFIG:-configs/random_search_uni_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
