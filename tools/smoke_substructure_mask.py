"""Smoke test: substructure mask semantics.

Validates that the ``substructure`` template mask:

  - in **uni mode** is bit-identical to the legacy (pre-fix) behaviour for
    every test molecule (R1 first-reactant substructure match only — no
    apply_reaction call, no R2 check),
  - in **bi mode** still passes uni-type templates by R1 match only, but
    additionally requires bi-type templates to have at least one R2 in the
    pool that pattern-matches the template's second reactant slot.

To keep the smoke fast we sample 20 molecules from each split and compare
against a hand-rolled reference implementation that reproduces the pre-fix
semantics. Run::

    PYTHONPATH=. python -u tools/smoke_substructure_mask.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from genmolrl.chem.datasets import load_pickle
from genmolrl.chem.reaction_manager import BI_TYPE, UNI_TYPES, ReactionManager
from genmolrl.config import resolve_path


def legacy_substructure_mask(mgr: ReactionManager, smiles: str) -> np.ndarray:
    """Pre-fix substructure mask: R1 first-reactant match only, no R2 check."""
    return np.array(
        [int(mgr.match_template(smiles, t)["first"]) for t in mgr.templates.values()],
        dtype=np.int8,
    )


def main() -> None:
    # ---- UNI ------------------------------------------------------------
    reactants_uni = load_pickle(Path(resolve_path("data/Uni/reactants_train.pkl")))
    test_uni = load_pickle(Path(resolve_path("data/Uni/reactants_test.pkl")))
    templates_uni_raw = load_pickle(
        Path(resolve_path("data/Uni/templates_unimolecolar_explicit.pkl"))
    )
    mgr_uni = ReactionManager(templates_uni_raw, reactants_uni)
    mgr_uni.templates = mgr_uni.templates_for_mode("uni")
    mgr_uni.template_keys = list(mgr_uni.templates.keys())
    mgr_uni.template_mask_cache.clear()

    uni_states = list(test_uni.keys())[:20]
    BI_N = 3  # bi check walks 116k reactants × 87 bi templates per state on cold cache, so keep small.
    print(f"[uni] checking substructure mask invariance over {len(uni_states)} states")
    n_t = len(mgr_uni.templates)
    print(f"[uni] #templates={n_t}")
    for state in uni_states:
        new_mask = mgr_uni.template_substructure_mask(state).cpu().numpy().astype(np.int8)
        legacy_mask = legacy_substructure_mask(mgr_uni, state)
        assert np.array_equal(new_mask, legacy_mask), (
            f"[uni] substructure mask CHANGED for state={state!r}; "
            f"diff at {np.where(new_mask != legacy_mask)[0].tolist()}"
        )
    print("[uni] OK: substructure mask is bit-identical to legacy for all sampled states.")

    # ---- BI -------------------------------------------------------------
    reactants_bi = load_pickle(Path(resolve_path("data/Bi/reactants_train.pkl")))
    test_bi = load_pickle(Path(resolve_path("data/Bi/reactants_test.pkl")))
    templates_bi_raw = load_pickle(Path(resolve_path("data/Bi/templates.pkl")))
    mgr_bi = ReactionManager(templates_bi_raw, reactants_bi)
    mgr_bi.templates = mgr_bi.templates_for_mode("bi")
    mgr_bi.template_keys = list(mgr_bi.templates.keys())
    mgr_bi.template_mask_cache.clear()
    bi_template_indices = [
        idx for idx, t in mgr_bi.templates.items() if t.get("type") == BI_TYPE
    ]
    uni_template_indices = [
        idx for idx, t in mgr_bi.templates.items() if t.get("type") in UNI_TYPES
    ]
    print(
        f"[bi] #templates={len(mgr_bi.templates)} (uni={len(uni_template_indices)} bi={len(bi_template_indices)})"
    )

    bi_states = list(test_bi.keys())[:BI_N]
    print(
        f"[bi] checking substructure mask semantics over {len(bi_states)} states "
        f"(pool scan is ~5min/state on cold cache; keeping count small)"
    )
    matched_uni_diff = 0
    matched_bi_diff = 0
    for state in bi_states:
        new_mask = mgr_bi.template_substructure_mask(state).cpu().numpy().astype(np.int8)
        legacy_mask = legacy_substructure_mask(mgr_bi, state)

        # Uni-type templates: NEW must equal LEGACY (R1 match only, no R2 check).
        for idx in uni_template_indices:
            if new_mask[idx] != legacy_mask[idx]:
                matched_uni_diff += 1
                raise AssertionError(
                    f"[bi] uni template {idx} mask changed for state={state!r}: "
                    f"legacy={legacy_mask[idx]} new={new_mask[idx]} (uni branch must be unchanged)"
                )

        # Bi-type templates: NEW = LEGACY ∧ (∃ R2 pattern match). NEW ≤ LEGACY.
        for idx in bi_template_indices:
            legacy = int(legacy_mask[idx])
            new = int(new_mask[idx])
            if legacy == 0:
                assert new == 0, (
                    f"[bi] bi template {idx} flipped 0→1 (new check is stricter, must not flip on)"
                )
                continue
            # legacy == 1: new is 1 iff there's at least one R2 with pattern match.
            has_r2 = bool(mgr_bi.get_valid_reactants(idx))
            expected_new = 1 if has_r2 else 0
            assert new == expected_new, (
                f"[bi] bi template {idx} mask wrong for state={state!r}: "
                f"legacy=1, has_r2={has_r2}, new={new}, expected={expected_new}"
            )
            if new != legacy:
                matched_bi_diff += 1
    print(
        f"[bi] OK: uni branch unchanged; bi branch correctly stricter "
        f"(bi-template positions changed on {matched_bi_diff} state×template events)."
    )

    # Spot-print one mask comparison for visual sanity.
    s = bi_states[0]
    new = mgr_bi.template_substructure_mask(s).cpu().numpy().astype(np.int8)
    legacy = legacy_substructure_mask(mgr_bi, s)
    print(
        f"\n[bi] state={s[:60]:<62s} legacy_T={int(legacy.sum())} new_T={int(new.sum())} "
        f"(of {len(new)} total templates)"
    )

    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
