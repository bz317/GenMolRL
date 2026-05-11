"""Lightweight benchmark: per-state true-validity R2 mask cost for Bi-PPO.

Avoids the expensive `get_valid_reactants` warm-up over all 87 bi-templates
(which iterates 116k reactants per template); instead we lazily evaluate the
mask only for the templates that the policy would actually sample for each
state (the ones flagged valid by `template_reaction_valid_mask`).

This mirrors what hierarchical Bi-PPO will do at runtime: for each state,
sample T from the valid template set (~7 bi templates), then compute the
true-valid R2 mask ONLY for that single T. The "warm-up" is therefore
amortised across episodes.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from genmolrl.chem.datasets import load_pickle
from genmolrl.chem.reaction_manager import BI_TYPE, ReactionManager
from genmolrl.config import resolve_path


def main() -> None:
    reactants = load_pickle(Path(resolve_path("data/Bi/reactants_train.pkl")))
    templates_raw = load_pickle(Path(resolve_path("data/Bi/templates.pkl")))
    test_pool = load_pickle(Path(resolve_path("data/Bi/reactants_test.pkl")))

    all_mgr = ReactionManager(templates_raw, reactants)
    bi_only = {i: t for i, t in all_mgr.templates.items() if t.get("type") == BI_TYPE}
    mgr = ReactionManager(bi_only, reactants)
    test_smiles = list(test_pool.keys())
    rng = random.Random(42)
    rng.shuffle(test_smiles)

    print(f"#bi-templates={len(mgr.templates)}  #reactants(pool)={len(mgr.reactants)}")
    print("(lazy: get_valid_reactants is only warmed for sampled templates)")

    per_state_mask_seconds: list[float] = []
    per_state_n_valid_t: list[int] = []
    per_state_pattern_total: list[int] = []
    per_state_true_valid_total: list[int] = []

    N_STATES = 30  # cheap pilot
    for s_i in range(N_STATES):
        smiles = test_smiles[s_i]
        t_start = time.monotonic()
        tmpl_mask = mgr.template_reaction_valid_mask(smiles).cpu().numpy()
        valid_t_idxs = [int(i) for i in np.where(tmpl_mask > 0.5)[0]]

        n_pattern = 0
        n_true_valid = 0
        for t_idx in valid_t_idxs:
            # Use the new ReactionManager helper directly so we benchmark the
            # exact code path the hierarchical Bi-PPO trainer will hit.
            mask = mgr.bi_r2_valid_mask(smiles, t_idx)
            pattern = mgr.get_valid_reactants(t_idx)
            n_pattern += len(pattern)
            n_true_valid += int(mask.sum())

        elapsed = time.monotonic() - t_start
        per_state_mask_seconds.append(elapsed)
        per_state_n_valid_t.append(len(valid_t_idxs))
        per_state_pattern_total.append(n_pattern)
        per_state_true_valid_total.append(n_true_valid)
        print(
            f"[{s_i:3d}] state={smiles[:50]:<52s} "
            f"valid_T={len(valid_t_idxs):>2}  "
            f"pattern_R2={n_pattern:>6}  true_valid_R2={n_true_valid:>6}  "
            f"true/pattern={(n_true_valid / n_pattern) if n_pattern else 0:.2%}  "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )

    print()
    print(f"=== Summary across {N_STATES} states ===")
    print(f"mean per-state cost: {np.mean(per_state_mask_seconds):.2f}s   median: {np.median(per_state_mask_seconds):.2f}s   max: {np.max(per_state_mask_seconds):.2f}s")
    print(f"mean #valid templates per state: {np.mean(per_state_n_valid_t):.2f}")
    print(f"mean pattern R2 per state:       {np.mean(per_state_pattern_total):.0f}")
    print(f"mean true-valid R2 per state:    {np.mean(per_state_true_valid_total):.0f}")
    p, v = sum(per_state_pattern_total), sum(per_state_true_valid_total)
    print(f"overall true-valid / pattern retention: {(v / p) if p else 0:.2%}")
    print(f"overall pattern -> -1 leak rate:        {(1 - v / p) if p else 0:.2%}")


if __name__ == "__main__":
    main()
