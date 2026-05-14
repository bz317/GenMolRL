#!/usr/bin/env bash
# Bi random-search baseline, train-R2 variant. Same algorithm as
# run_genmolrl_random_search_bi.sh — one uniformly-random valid (T, R2)
# pair per step, up to max_episode_len steps, under r2_available masking
# — but the R2 candidate pool is the *training* reactant pool while the
# start molecules stay on the *test* pool (the start_pool / r2_pool split
# is owned by configs/random_search_bi_train_r2_delta_qed.yaml via
# dataset.r2_pool_file). This is the matched random baseline for any
# bi-PPO eval loop that runs with r2_arch: lookup + eval_r2_pool: train
# (the gr7aa7z6 setup).
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
  --config "${CONFIG:-configs/random_search_bi_train_r2_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
