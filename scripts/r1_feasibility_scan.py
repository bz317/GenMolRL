#!/usr/bin/env python
"""R1-only template feasibility scan over four reactant sets.

For each molecule in each input set, count how many reaction templates the
molecule satisfies as the first reactant (R1) under an RDKit substructure
match. This is the same "R1 substructure" criterion that
`ReactionManager.template_substructure_mask` uses (`useChirality=True`) but
applied directly without running the full reaction (which is the slow part of
the strict feasibility check).

Inputs (hard-coded, per user request):
    1. designing-new-molecules/data/new_data/reactants_train.pkl
    2. designing-new-molecules/data/new_data/reactants_val.pkl
    3. designing-new-molecules/data/train_test_split/uni_molecular/reactants_train.pkl
    4. designing-new-molecules/data/train_test_split/uni_molecular/reactants_val.pkl

Templates: designing-new-molecules/data/train_test_split/bi_molecular/templates.pkl
(102 templates; superset of unimolecular + bimolecular).

Outputs (saved to GenMolRL/data/train_test_split/):
    * feasibility_summary.csv          # per-set stats over all 102 templates
    * feasibility_summary_by_type.csv  # per-set stats split into uni (15) / bi (87)
    * feasibility_options.csv          # Option-1 (new_data train/val) vs Option-2
                                       #   (uni/val test, train pool = new_data \\ uni/val)
    * feasibility_counts.npz           # per-pool int32 #matched templates (all + uni + bi)
    * feasibility_template_hits.csv    # per-template hit counts in each pool
    * feasibility_distribution.png     # all-templates histogram + boxplot (4 sets)
    * feasibility_distribution_by_type.png  # uni and bi histograms per pool
    * feasibility_options.png          # Option-1 vs Option-2 train/test comparison
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path("/home/bz317/new_mol_with_RL_GNN")
TEMPLATES_PKL = REPO_ROOT / "designing-new-molecules/data/train_test_split/bi_molecular/templates.pkl"

SETS = {
    "new_data/train": REPO_ROOT / "designing-new-molecules/data/new_data/reactants_train.pkl",
    "new_data/val": REPO_ROOT / "designing-new-molecules/data/new_data/reactants_val.pkl",
    "uni/train": REPO_ROOT / "designing-new-molecules/data/train_test_split/uni_molecular/reactants_train.pkl",
    "uni/val": REPO_ROOT / "designing-new-molecules/data/train_test_split/uni_molecular/reactants_val.pkl",
}

OUT_DIR = REPO_ROOT / "GenMolRL/data/train_test_split"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_smiles_list(p: Path) -> list[str]:
    """Reactant pickles in this repo are `dict[smiles -> fingerprint]`."""
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        smiles = [str(k) for k in obj.keys()]
    elif isinstance(obj, list):
        smiles = [str(x) for x in obj]
    else:
        raise TypeError(f"Unknown pickle type for {p}: {type(obj)}")
    return smiles


def load_templates(p: Path) -> list[dict]:
    with open(p, "rb") as f:
        obj = pickle.load(f)
    out = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                out.append(item[1])
            elif isinstance(item, dict):
                out.append(item)
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, dict):
                out.append(v)
    return out


def template_smarts(t: dict) -> str:
    return t.get("smarts") or t.get("_original_smarts")


def first_reactant_smarts(t: dict) -> str | None:
    """Return the SMARTS string of the *first* reactant on the LHS of the template.

    We deliberately re-extract by string splitting and wrap with `Chem.MolFromSmarts`
    rather than reuse `rxn.GetReactantTemplate(0)`, because the latter sometimes
    segfaults under `HasSubstructMatch` on recursive SMARTS like `[$(...),CH:10]`.
    """
    s = template_smarts(t)
    if not s or ">>" not in s:
        return None
    lhs = s.split(">>", 1)[0]
    if "." in lhs:
        return lhs.split(".", 1)[0]
    return lhs


def first_reactant_pattern(t: dict):
    smarts = first_reactant_smarts(t)
    if not smarts:
        return None
    try:
        return Chem.MolFromSmarts(smarts)
    except Exception:
        return None


def _is_uni(template: dict) -> bool:
    """Treat any `_original_type=='unimolecular'` (or `type` containing 'unimolecular')
    as a unimolecular template. Bimolecular templates have type=='bimolecular'."""
    orig = template.get("_original_type") or template.get("type") or ""
    return "unimolecular" in str(orig).lower() and "bimolecular" not in str(orig).lower()


def _stats_block(arr: np.ndarray, label: str) -> dict:
    n = len(arr)
    if n == 0:
        return {"set": label, "n": 0}
    return {
        "set": label,
        "n": n,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "frac_>=1": float((arr >= 1).sum()) / n,
        "frac_>=5": float((arr >= 5).sum()) / n,
        "frac_>=10": float((arr >= 10).sum()) / n,
    }


def main() -> None:
    print(f"Loading templates: {TEMPLATES_PKL}")
    templates = load_templates(TEMPLATES_PKL)
    n_templates = len(templates)
    is_uni = np.array([_is_uni(t) for t in templates], dtype=bool)
    is_bi = ~is_uni
    print(f"  loaded {n_templates} templates ({int(is_uni.sum())} uni, {int(is_bi.sum())} bi)")

    template_patterns: list = []
    for t in templates:
        template_patterns.append(first_reactant_pattern(t))
    n_valid = sum(p is not None for p in template_patterns)
    print(f"  {n_valid}/{n_templates} templates yielded a valid R1 pattern")

    set_smiles: dict[str, list[str]] = {}
    for name, path in SETS.items():
        print(f"Loading {name}: {path}")
        smiles = load_smiles_list(path)
        set_smiles[name] = smiles
        print(f"  loaded {len(smiles)} smiles")

    nd_train = set(set_smiles["new_data/train"])
    nd_val = set(set_smiles["new_data/val"])
    uni_val = set(set_smiles["uni/val"])
    uni_train = set(set_smiles["uni/train"])

    print("\nOverlap diagnostics:")
    print(f"  |new_data/train ∩ new_data/val|       = {len(nd_train & nd_val)}")
    print(f"  |new_data/train ∩ uni/val|            = {len(nd_train & uni_val)}")
    print(f"  |new_data/val   ∩ uni/val|            = {len(nd_val & uni_val)}")
    print(f"  |new_data ∪ ∪new_data/val ∩ uni/val|  = {len((nd_train | nd_val) & uni_val)}")
    print(f"  |uni/val \\ new_data|                  = {len(uni_val - (nd_train | nd_val))}")
    print(f"  |uni/train ∩ new_data|                = {len(uni_train & (nd_train | nd_val))}")

    opt2_train_pool: list[str] = sorted((nd_train | nd_val) - uni_val)
    set_smiles["opt2_train_pool"] = opt2_train_pool
    print(f"\nOption-2 training pool (= new_data ∪ new_data/val − uni/val): "
          f"{len(opt2_train_pool)} smiles")

    all_smiles: set[str] = set()
    for smiles in set_smiles.values():
        all_smiles.update(smiles)
    unique_smiles = sorted(all_smiles)
    print(f"Total unique SMILES across all pools: {len(unique_smiles)}")

    print("\nParsing unique SMILES → RDKit Mol …")
    t0 = time.time()
    smi_to_mol: dict[str, object] = {}
    n_failed = 0
    for s in unique_smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            n_failed += 1
            continue
        smi_to_mol[s] = m
    print(f"  parsed in {time.time()-t0:.1f}s ({n_failed} failed)")

    print("\nRunning R1 substructure scan (useChirality=True)…")
    t0 = time.time()
    smiles_list = list(smi_to_mol.keys())
    n_unique = len(smiles_list)
    hit_matrix = np.zeros((n_unique, n_templates), dtype=np.int8)
    smi_index = {s: i for i, s in enumerate(smiles_list)}

    for j, pat in enumerate(template_patterns):
        t_start = time.time()
        if pat is None:
            print(f"  template {j+1:3d}/{n_templates} | SKIP (no R1 pattern)", flush=True)
            continue
        for i, s in enumerate(smiles_list):
            mol = smi_to_mol[s]
            try:
                if mol.HasSubstructMatch(pat, useChirality=True):
                    hit_matrix[i, j] = 1
            except Exception:
                pass
        if (j + 1) % 10 == 0 or j == n_templates - 1 or j < 2:
            elapsed = time.time() - t0
            done_frac = (j + 1) / n_templates
            eta = elapsed / max(done_frac, 1e-6) - elapsed
            tname = templates[j].get("name", "")[:30]
            print(
                f"  template {j+1:3d}/{n_templates} ({tname:30s}) "
                f"| dt={time.time()-t_start:5.1f}s | elapsed={elapsed:6.1f}s | eta={eta:6.1f}s",
                flush=True,
            )
    print(f"  scan done in {time.time()-t0:.1f}s")

    counts_total = hit_matrix.sum(axis=1).astype(np.int32)
    counts_uni = hit_matrix[:, is_uni].sum(axis=1).astype(np.int32)
    counts_bi = hit_matrix[:, is_bi].sum(axis=1).astype(np.int32)

    set_counts_all: dict[str, np.ndarray] = {}
    set_counts_uni: dict[str, np.ndarray] = {}
    set_counts_bi: dict[str, np.ndarray] = {}
    set_template_hits: dict[str, np.ndarray] = {}
    for name, smiles in set_smiles.items():
        idxs = np.array([smi_index[s] for s in smiles if s in smi_index])
        set_counts_all[name] = counts_total[idxs]
        set_counts_uni[name] = counts_uni[idxs]
        set_counts_bi[name] = counts_bi[idxs]
        set_template_hits[name] = hit_matrix[idxs].sum(axis=0).astype(np.int64)

    rows_all = []
    rows_by_type = []
    for name in set_smiles:
        arr_all = set_counts_all[name]
        arr_uni = set_counts_uni[name]
        arr_bi = set_counts_bi[name]
        n = len(arr_all)
        nonzero_all = int((arr_all > 0).sum())
        rows_all.append({
            "set": name,
            "n_molecules": n,
            "mean_templates": float(arr_all.mean()),
            "median_templates": float(np.median(arr_all)),
            "std_templates": float(arr_all.std()),
            "min_templates": int(arr_all.min()),
            "max_templates": int(arr_all.max()),
            "p25": float(np.percentile(arr_all, 25)),
            "p75": float(np.percentile(arr_all, 75)),
            "p95": float(np.percentile(arr_all, 95)),
            "frac_with_any_template": nonzero_all / max(n, 1),
            "frac_with_5+_templates": float((arr_all >= 5).sum()) / max(n, 1),
            "frac_with_10+_templates": float((arr_all >= 10).sum()) / max(n, 1),
        })
        for type_label, arr in [("uni", arr_uni), ("bi", arr_bi)]:
            rows_by_type.append({
                "set": name,
                "template_type": type_label,
                "n_templates_in_type": int(is_uni.sum() if type_label == "uni" else is_bi.sum()),
                "n_molecules": n,
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "frac_>=1": float((arr >= 1).sum()) / max(n, 1),
                "frac_>=3": float((arr >= 3).sum()) / max(n, 1),
                "frac_>=5": float((arr >= 5).sum()) / max(n, 1),
            })

    summary = pd.DataFrame(rows_all)
    summary_path = OUT_DIR / "feasibility_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary (all templates) written to {summary_path}")
    print(summary.to_string(index=False))

    summary_by_type = pd.DataFrame(rows_by_type)
    summary_by_type_path = OUT_DIR / "feasibility_summary_by_type.csv"
    summary_by_type.to_csv(summary_by_type_path, index=False)
    print(f"\nSummary (split by template type) written to {summary_by_type_path}")
    print(summary_by_type.to_string(index=False))

    options_rows = []
    pairs = [
        ("Option_1", "train", "new_data/train"),
        ("Option_1", "test",  "new_data/val"),
        ("Option_2", "train", "opt2_train_pool"),
        ("Option_2", "test",  "uni/val"),
    ]
    for opt, role, name in pairs:
        for type_label, counts_dict, n_in_type in [
            ("all", set_counts_all, n_templates),
            ("uni", set_counts_uni, int(is_uni.sum())),
            ("bi",  set_counts_bi,  int(is_bi.sum())),
        ]:
            arr = counts_dict[name]
            options_rows.append({
                "option": opt,
                "role": role,
                "set": name,
                "template_type": type_label,
                "n_templates_in_type": n_in_type,
                "n_molecules": len(arr),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "frac_>=1": float((arr >= 1).sum()) / max(len(arr), 1),
                "frac_>=3": float((arr >= 3).sum()) / max(len(arr), 1),
            })
    options_df = pd.DataFrame(options_rows)
    options_path = OUT_DIR / "feasibility_options.csv"
    options_df.to_csv(options_path, index=False)
    print(f"\nOption-1 vs Option-2 stats written to {options_path}")
    print(options_df.to_string(index=False))

    npz_payload: dict[str, np.ndarray] = {}
    for name, arr in set_counts_all.items():
        npz_payload[f"{name}|all"] = arr
        npz_payload[f"{name}|uni"] = set_counts_uni[name]
        npz_payload[f"{name}|bi"] = set_counts_bi[name]
    np.savez_compressed(OUT_DIR / "feasibility_counts.npz", **npz_payload)
    print(f"\nPer-molecule counts: {OUT_DIR/'feasibility_counts.npz'}")

    template_hit_df = pd.DataFrame(set_template_hits)
    template_hit_df.index.name = "template_idx"
    template_names = [t.get("name", "") for t in templates]
    template_types = [t.get("type", "") for t in templates]
    template_hit_df.insert(0, "template_name", template_names)
    template_hit_df.insert(1, "template_type", template_types)
    template_hit_df.to_csv(OUT_DIR / "feasibility_template_hits.csv")
    print(f"Per-template hit counts: {OUT_DIR/'feasibility_template_hits.csv'}")

    plot_4set_overall(set_counts_all)
    plot_by_type(set_counts_uni, set_counts_bi, n_uni=int(is_uni.sum()), n_bi=int(is_bi.sum()))
    plot_options(set_counts_uni, set_counts_bi, set_counts_all)


def plot_4set_overall(set_counts_all: dict[str, np.ndarray]) -> None:
    pools = ["new_data/train", "new_data/val", "uni/train", "uni/val"]
    counts = {p: set_counts_all[p] for p in pools}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    max_count = int(max(arr.max() for arr in counts.values()))

    ax = axes[0]
    bins = np.arange(0, max_count + 2)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, arr), c in zip(counts.items(), colors):
        ax.hist(arr, bins=bins, histtype="step", density=True, linewidth=1.8,
                label=f"{name} (n={len(arr)}, mean={arr.mean():.2f})", color=c)
    ax.set_xlabel("# templates with R1 substructure match (out of 102)")
    ax.set_ylabel("Density")
    ax.set_title("Per-molecule R1 feasibility distribution")
    ax.set_xlim(0, max_count + 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    data = [counts[name] for name in pools]
    box = ax.boxplot(data, tick_labels=pools, showmeans=True, showfliers=False, patch_artist=True)
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.4)
    ax.set_ylabel("# templates with R1 substructure match")
    ax.set_title("Boxplot per set (whiskers exclude outliers)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("R1-only template feasibility (RDKit substructure, useChirality=True, 102 templates)")
    fig.tight_layout()
    out_png = OUT_DIR / "feasibility_distribution.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Distribution plot: {out_png}")


def plot_by_type(set_counts_uni: dict, set_counts_bi: dict, *, n_uni: int, n_bi: int) -> None:
    pools = ["new_data/train", "new_data/val", "uni/train", "uni/val"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (counts_dict, type_label, n_in_type) in zip(
        axes,
        [
            (set_counts_uni, f"uni-templates ({n_uni})", n_uni),
            (set_counts_bi,  f"bi-templates ({n_bi})",  n_bi),
        ],
    ):
        max_count = int(max(counts_dict[p].max() for p in pools))
        bins = np.arange(0, max_count + 2)
        for p, c in zip(pools, colors):
            arr = counts_dict[p]
            ax.hist(arr, bins=bins, histtype="step", density=True, linewidth=1.8,
                    label=f"{p} (mean={arr.mean():.2f})", color=c)
        ax.set_xlabel(f"# {type_label} matched as R1")
        ax.set_ylabel("Density")
        ax.set_title(f"R1 feasibility on {type_label} (out of {n_in_type})")
        ax.set_xlim(0, max_count + 1)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("R1-only template feasibility split by template type")
    fig.tight_layout()
    out_png = OUT_DIR / "feasibility_distribution_by_type.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"By-type distribution plot: {out_png}")


def plot_options(set_counts_uni: dict, set_counts_bi: dict, set_counts_all: dict) -> None:
    """Compare Option-1 (new_data train/val) vs Option-2 (opt2 train pool / uni/val)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (counts_dict, type_label) in zip(
        axes,
        [(set_counts_all, "all 102"),
         (set_counts_uni, "uni-only (15)"),
         (set_counts_bi,  "bi-only (87)")],
    ):
        opt1_train = counts_dict["new_data/train"]
        opt1_test  = counts_dict["new_data/val"]
        opt2_train = counts_dict["opt2_train_pool"]
        opt2_test  = counts_dict["uni/val"]
        data = [opt1_train, opt1_test, opt2_train, opt2_test]
        labels = [
            f"O1 train\nnew_data/train\nμ={opt1_train.mean():.2f}",
            f"O1 test\nnew_data/val\nμ={opt1_test.mean():.2f}",
            f"O2 train\nnd \\ uni/val\nμ={opt2_train.mean():.2f}",
            f"O2 test\nuni/val\nμ={opt2_test.mean():.2f}",
        ]
        colors = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"]
        box = ax.boxplot(
            data, tick_labels=labels, showmeans=True, showfliers=False, patch_artist=True,
        )
        for patch, c in zip(box["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_ylabel(f"# {type_label} templates matched as R1")
        ax.set_title(f"Option 1 vs Option 2 — {type_label}")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), fontsize=8)

    fig.suptitle(
        "Train/test feasibility match: "
        "Option 1 = new_data train→val, "
        "Option 2 = (new_data ∪ new_data/val) \\ uni/val → uni/val",
    )
    fig.tight_layout()
    out_png = OUT_DIR / "feasibility_options.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Options comparison plot: {out_png}")


if __name__ == "__main__":
    main()
