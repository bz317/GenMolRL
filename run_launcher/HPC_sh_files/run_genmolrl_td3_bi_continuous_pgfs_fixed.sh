#!/usr/bin/env bash
# Bi-TD3 PGFS-faithful with the three Bi-TD3-only code fixes:
#   (A) knn.py:    drop the (vec >= 0).float() binariser; rescale FAISS keys
#                  from {0, 1} → {-1, +1}.
#   (B) train.py:  rescale warm-up R(2) FP from {0, 1} → {-1, +1} so warm-up
#                  and actor outputs share the tanh range (replay
#                  distribution-shift fix).
#   (C) models.py: drop the trailing 167-dim bottleneck in PiNetwork (Bi-TD3
#                  is the only consumer of this layer).
# All hyperparameters / experiment name come from the YAML config below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "${ROOT}/run_genmolrl_td3_common.sh" \
  "configs/td3_bi_continuous_pgfs_fixed_delta_qed.yaml"
