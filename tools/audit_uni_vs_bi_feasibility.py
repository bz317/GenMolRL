"""Side-by-side R1-feasibility for uni and bi templates across the test set."""

from __future__ import annotations

import os
import time

os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np  # noqa: E402

from genmolrl.chem.datasets import load_pickle  # noqa: E402
from genmolrl.chem.reaction_manager import ReactionManager  # noqa: E402
from genmolrl.config import load_config, resolve_path  # noqa: E402


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _stats(name: str, arr: np.ndarray, bins: list[int]) -> None:
    n_total = len(arr)
    _print(f"  ---- {name} ----")
    _print(f"  n molecules                    : {n_total}")
    _print(f"  molecules with 0 R1-feasible   : {int((arr == 0).sum())}")
    _print(f"  molecules with ≥1 R1-feasible  : {int((arr > 0).sum())}")
    _print(f"  mean                           : {arr.mean():.4f}")
    _print(f"  median                         : {int(np.median(arr))}")
    _print(f"  min / max                      : {int(arr.min())} / {int(arr.max())}")
    _print(f"  std                            : {arr.std():.4f}")
    _print(
        f"  percentiles 25/50/75/90/99     : "
        f"{int(np.percentile(arr, 25))} / "
        f"{int(np.percentile(arr, 50))} / "
        f"{int(np.percentile(arr, 75))} / "
        f"{int(np.percentile(arr, 90))} / "
        f"{int(np.percentile(arr, 99))}"
    )
    _print("  histogram:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = int(((arr >= lo) & (arr < hi)).sum())
        bar = "#" * int(60 * in_bin / max(1, n_total))
        _print(f"    [{lo:>4d}, {hi:>4d}): {in_bin:>6d}  {bar}")
    in_bin = int((arr >= bins[-1]).sum())
    _print(f"    [{bins[-1]:>4d},   ∞): {in_bin:>6d}")


def main() -> None:
    cfg_bi = load_config("configs/exhausted_search_bi_delta_qed.yaml")
    cfg_uni = load_config("configs/exhausted_search_uni_delta_qed.yaml")

    uni_templates_pkl = load_pickle(resolve_path(cfg_uni["dataset"]["templates_file"]))
    bi_templates_pkl = load_pickle(resolve_path(cfg_bi["dataset"]["templates_file"]))
    uni_test = load_pickle(resolve_path(cfg_uni["dataset"]["test_file"]))
    bi_test = load_pickle(resolve_path(cfg_bi["dataset"]["test_file"]))

    _print(f"uni templates pickle : {cfg_uni['dataset']['templates_file']}  "
           f"({len(uni_templates_pkl)} entries)")
    _print(f"bi  templates pickle : {cfg_bi['dataset']['templates_file']}  "
           f"({len(bi_templates_pkl)} entries)")
    _print(f"uni test reactants   : {cfg_uni['dataset']['test_file']}  ({len(uni_test)} mols)")
    _print(f"bi  test reactants   : {cfg_bi['dataset']['test_file']}  ({len(bi_test)} mols)")

    mgr_uni_full = ReactionManager(uni_templates_pkl, uni_test)
    uni_only = mgr_uni_full.templates_for_mode("uni")
    mgr_uni = ReactionManager(uni_only, uni_test)
    _print(f"uni templates after mode filter : {len(uni_only)}")

    mgr_bi_full = ReactionManager(bi_templates_pkl, bi_test)
    bi_only = mgr_bi_full.templates_for_mode("bi")
    mgr_bi = ReactionManager(bi_only, bi_test)
    _print(f"bi  templates after mode filter : {len(bi_only)}")

    # The `reaction_valid` mask is uni-only by design (see ReactionManager.
    # template_reaction_valid_mask) -- it requires apply_reaction(s, t, None)
    # to succeed, which is impossible for genuine bi templates. For bi-mode
    # feasibility we must use `r2_available` (alias `executable`), which
    # checks R1 + ≥1 valid R2 in the bank. We compute both masks for both
    # template pickles so the asymmetry between them is visible.

    def _sweep(mgr, keys, kind, label):
        arr = np.zeros(len(keys), dtype=np.int32)
        progress = max(1, len(keys) // 5)
        t0 = time.time()
        for i, s in enumerate(keys):
            try:
                arr[i] = len(mgr.feasible_first_reactant_templates(s, kind=kind))
            except Exception:
                arr[i] = -1
            if (i + 1) % progress == 0:
                elapsed = time.time() - t0
                est = elapsed * len(keys) / (i + 1)
                _print(f"  [{label}] progress: {i + 1}/{len(keys)}  elapsed={elapsed:.1f}s  est={est:.0f}s")
        _print(f"  [{label}] finished in {time.time() - t0:.1f}s")
        return arr[arr >= 0]

    keys_uni = list(uni_test.keys())
    keys_bi = list(bi_test.keys())

    _print(f"\n[1/4] uni templates × uni test, kind='reaction_valid'")
    arr_uni_rxn = _sweep(mgr_uni, keys_uni, "reaction_valid", "uni-rxn")

    _print(f"\n[2/4] bi  templates × bi  test, kind='reaction_valid' (legacy mask — uni-only)")
    arr_bi_rxn = _sweep(mgr_bi, keys_bi, "reaction_valid", "bi-rxn")

    _print(f"\n[3/4] uni templates × uni test, kind='r2_available' (correct bi mask)")
    arr_uni_r2 = _sweep(mgr_uni, keys_uni, "r2_available", "uni-r2")

    _print(f"\n[4/4] bi  templates × bi  test, kind='r2_available' (correct bi mask)")
    arr_bi_r2 = _sweep(mgr_bi, keys_bi, "r2_available", "bi-r2")

    _print("\n========== RESULTS ==========")
    _stats(f"UNI templates (n={len(uni_only)}) — kind=reaction_valid",
           arr_uni_rxn, bins=[0, 1, 2, 3, 5, 10, 15])
    _print()
    _stats(f"UNI templates (n={len(uni_only)}) — kind=r2_available",
           arr_uni_r2, bins=[0, 1, 2, 3, 5, 10, 15])
    _print()
    _stats(f"BI  templates (n={len(bi_only)}) — kind=reaction_valid (LEGACY, uni-only)",
           arr_bi_rxn, bins=[0, 1, 2, 3, 5, 10, 20, 40, 80, 102])
    _print()
    _stats(f"BI  templates (n={len(bi_only)}) — kind=r2_available (CORRECT)",
           arr_bi_r2, bins=[0, 1, 2, 3, 5, 10, 20, 40, 80, 102])

    _print("\nHeadline numbers (mean ± std):")
    _print(f"  uni feasibility (reaction_valid mask) : {arr_uni_rxn.mean():.4f} ± {arr_uni_rxn.std():.4f}")
    _print(f"  uni feasibility (r2_available  mask)  : {arr_uni_r2.mean():.4f}  ± {arr_uni_r2.std():.4f}")
    _print(f"  bi  feasibility (reaction_valid mask) : {arr_bi_rxn.mean():.4f} ± {arr_bi_rxn.std():.4f}    (under-counts!)")
    _print(f"  bi  feasibility (r2_available  mask)  : {arr_bi_r2.mean():.4f}  ± {arr_bi_r2.std():.4f}")

    if keys_uni == keys_bi:
        _print("\n(uni and bi test pickles contain the same SMILES in the same order)")
        combined = arr_uni_r2 + arr_bi_r2
        _print(f"\nPer-molecule UNI + BI total R1-feasible templates "
               f"(using r2_available for both): mean={combined.mean():.4f}, "
               f"median={int(np.median(combined))}, max={int(combined.max())}, "
               f"#zero={int((combined == 0).sum())}")


if __name__ == "__main__":
    main()
