#!/usr/bin/env bash
# Common prelude sourced by every run_genmolrl_*.sh launcher.
#
# Responsibilities:
#   1. Resolve PROJECT_ROOT (the directory containing `genmolrl/`, `configs/`,
#      `data/`, `runs/`, `run_launcher/`) from this script's own location and
#      `cd` to it. Every launcher invokes python from PROJECT_ROOT so
#      `genmolrl.config.project_root()` and all `data/...`, `configs/...`,
#      `runs/...` paths in the YAMLs resolve correctly.
#   2. Activate the conda environment (defaults to `RL_for_new_mol`).
#   3. Export `WANDB_API_KEY` from a file OUTSIDE the project. The file lives
#      at the repo root, one level above PROJECT_ROOT, so it is never tracked
#      by the project's git repo. The `WANDB_API_KEY_FILE` env var can
#      override the path. If `WANDB_API_KEY` is already exported, we skip.
#   4. Build a `RUN_EXPERIMENT_OPT_ARGS` array of optional `--flag value`
#      pairs, ONE PAIR PER ENV VAR. The rule is: **only pass the flag when
#      the env var is explicitly set**, so the YAML config is the single
#      source of truth and `${VAR:-default}` cannot silently override it.
#
# Per-launcher: each launcher sources this file, then runs
#   PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m \
#     genmolrl.scripts.run_experiment \
#     --algorithm <X> --config <configs/...yaml> \
#     "${RUN_EXPERIMENT_OPT_ARGS[@]}"
#
# Source this file (do not exec it directly):
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_launcher_common.sh"

set -euo pipefail

# 1. Project root resolution -------------------------------------------------
_LAUNCHER_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_LAUNCHER_COMMON_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 2. Conda --------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-RL_for_new_mol}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# 3. wandb_api_key -----------------------------------------------------------
# Secrets must NEVER live inside the project. The default location is the
# parent of the project root (the outer repo dir). `WANDB_API_KEY` already
# being set takes priority; `WANDB_API_KEY_FILE` lets the user point to any
# absolute path on a per-invocation basis.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  _default_key_file="${PROJECT_ROOT}/../wandb_api_key.txt"
  _key_file="${WANDB_API_KEY_FILE:-${_default_key_file}}"
  if [[ -f "${_key_file}" ]]; then
    export WANDB_API_KEY="$(tr -d '\n\r ' < "${_key_file}")"
  fi
  unset _default_key_file _key_file
fi

# 4. Optional CLI overrides --------------------------------------------------
# Only forward a flag when the user explicitly set the env var. This keeps
# the YAML config authoritative; nothing here defaults to a hardcoded value.
RUN_EXPERIMENT_OPT_ARGS=()
if [[ -n "${REACTION_MODE:-}" ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--reaction-mode "${REACTION_MODE}"); fi
if [[ -n "${MASKING:-}"        ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--masking        "${MASKING}");        fi
if [[ -n "${REWARD:-}"         ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--reward         "${REWARD}");         fi
if [[ -n "${EXPERIMENT_NAME:-}" ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--experiment-name "${EXPERIMENT_NAME}"); fi
if [[ -n "${MAX_EPISODE_LEN:-}" ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--max-episode-len "${MAX_EPISODE_LEN}"); fi
if [[ -n "${TRAINING_FILE:-}"  ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--training-file  "${TRAINING_FILE}");  fi
if [[ -n "${TEST_FILE:-}"      ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--test-file      "${TEST_FILE}");      fi
# `TEMPLATES_FILE` is the canonical name; `TEMPLATE_FILE` is an older alias.
_TEMPLATES_FILE="${TEMPLATES_FILE:-${TEMPLATE_FILE:-}}"
if [[ -n "${_TEMPLATES_FILE}" ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--templates-file "${_TEMPLATES_FILE}"); fi
unset _TEMPLATES_FILE
if [[ -n "${GREEDY_MODE:-}"    ]]; then RUN_EXPERIMENT_OPT_ARGS+=(--greedy-mode    "${GREEDY_MODE}");    fi

# `STAGE_DATA` (true/false) is a launcher-level flag, not a YAML setting.
# Most launchers stage by default; this helper just exposes a uniform check.
should_stage_data() {
  [[ "${STAGE_DATA:-true}" == "true" ]]
}
