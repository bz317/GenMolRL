#!/usr/bin/env bash
# Bi random-search baseline. For each test start molecule the runner
# samples a single uniformly-random valid template (under r2_available
# masking) and then a single uniformly-random R2 from that template's
# pattern-match pool, applies the reaction, and repeats up to
# max_episode_len steps (or until no valid templates remain). RDKit
# sanitisation failures surface as a -1 reward and end the trajectory,
# matching the bi-PPO r2_available contract so the baseline is directly
# comparable. All hyperparameters / masking / reward / experiment name
# come from the YAML config below.
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
  --config "${CONFIG:-configs/random_search_bi_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
