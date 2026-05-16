#!/usr/bin/env bash
# Bi PPO hierarchical policy with legacy R2 lookup (r2_arch: lookup,
# eval_r2_pool: train — gr7aa7z6-style baseline). Same trainer as
# run_genmolrl_ppo_bi_hierarchical.sh; only the default YAML differs.
# Override CONFIG=... if you need a tweaked lookup YAML.
source "$(cd "$(dirname "$0")" && pwd)/_launcher_common.sh"

if should_stage_data; then
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python -m genmolrl.scripts.stage_data >/dev/null
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python -m genmolrl.scripts.run_experiment \
  --algorithm ppo_bi \
  --config "${CONFIG:-configs/ppo_bi_hierarchical_lookup_delta_qed.yaml}" \
  "${RUN_EXPERIMENT_OPT_ARGS[@]}"
