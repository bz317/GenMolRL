#!/usr/bin/env bash
# Uni GraphTransRL (trajectory balance) over a GraphTransformer backbone.
# All hyperparameters come from the YAML config below; export env vars only
# when you want to override the YAML at launch time.
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
  --algorithm graphtransrl \
  --config "${CONFIG:-configs/graphtransrl_uni_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
