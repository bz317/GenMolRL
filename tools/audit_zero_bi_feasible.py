"""Investigate the 5 test-set molecules that have 0 R1-feasible bi templates.

We already know from the prior full sweep that exactly 5 molecules in
data/Bi/reactants_test.pkl yield an empty
`feasible_first_reactant_templates(...)` list. This script:

  1. Hard-codes those 5 SMILES (so we don't re-sweep 12,689 mols).
  2. Cross-checks each against the uni template bank
     (data/Uni/templates_unimolecolar_explicit.pkl).
  3. Inspects the bi templates pickle structure and prints the R1 SMARTS
     patterns so we can see what the template bank actually covers.
  4. Reports the molecules' functional groups (using RDKit) to explain
     why they don't match.
"""

from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "disabled")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from genmolrl.chem.datasets import load_pickle  # noqa: E402
from genmolrl.chem.reaction_manager import BI_TYPE, ReactionManager  # noqa: E402
from genmolrl.config import load_config, resolve_path  # noqa: E402


ZERO_FEASIBLE_BI = [
    "CC(C)(C)OC(=O)N1[C@H]2CC[C@@H]1CC(=O)[C@H]2Br",
    "CC(=O)CC(=O)Nc1cccc(C(F)(F)F)c1",
    "COC(=O)c1ccccc1NC(=O)CC(C)=O",
    "COc1cccc(NC(=O)CC(C)=O)c1",
    "CC(=O)CC(=O)Nc1ccc([N+](=O)[O-])cc1",
]


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _describe_template(t) -> dict:
    """Normalise a template entry into a dict regardless of how the pickle stores it."""
    if isinstance(t, dict):
        return t
    return {"_raw": repr(t)[:80], "type": "?"}


def _functional_groups(smi: str) -> dict[str, int]:
    """Crude functional-group counts using SMARTS."""
    patterns = {
        "Primary amine (NH2)": "[NX3;H2;!$(NC=O)]",
        "Secondary amine (NHR)": "[NX3;H1;!$(NC=O)]",
        "Amide (C(=O)N)": "[CX3](=O)[NX3]",
        "Methyl ketone (CH3-CO-R)": "[CH3][CX3](=O)[#6]",
        "β-keto amide (O=C-C-C=O-N)": "[CX3](=O)[CX4][CX3](=O)[NX3]",
        "Ester (C(=O)O-R)": "[CX3](=O)[OX2][#6]",
        "Carboxylic acid": "[CX3](=O)[OX1H1]",
        "Boc carbamate (tBu-O-C(=O)-N)": "C(C)(C)(C)OC(=O)N",
        "Aryl bromide (Ar-Br)": "[c][Br]",
        "Alkyl bromide (C-Br)": "[CX4][Br]",
        "Aldehyde (CHO)": "[CX3H1](=O)",
        "Aniline (Ar-NH2)": "[NX3;H2][c]",
        "Aniline-amide (Ar-NHC(=O))": "[c][NX3;H1]C(=O)",
        "Nitro": "[NX3](=O)=O",
        "Trifluoromethyl (CF3)": "C(F)(F)F",
        "Aromatic alcohol (Ar-OH)": "[OX2H][c]",
        "Aliphatic alcohol (R-OH)": "[OX2H][CX4]",
    }
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    out: dict[str, int] = {}
    for name, smarts in patterns.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = len(mol.GetSubstructMatches(patt))
        if matches:
            out[name] = matches
    return out


def main() -> None:
    _print("[1/4] Cross-check the 5 zero-feasible bi molecules against UNI templates")
    cfg_uni = load_config("configs/exhausted_search_uni_delta_qed.yaml")
    uni_templates_pkl = load_pickle(resolve_path(cfg_uni["dataset"]["templates_file"]))
    uni_reactants = load_pickle(resolve_path(cfg_uni["dataset"]["test_file"]))
    _print(f"   loaded {len(uni_templates_pkl)} uni template entries, "
           f"{len(uni_reactants)} uni test reactants")
    mgr_uni = ReactionManager(uni_templates_pkl, uni_reactants)
    uni_only = mgr_uni.templates_for_mode("uni")
    _print(f"   uni-mode templates after filter: {len(uni_only)}")
    mgr_uni_f = ReactionManager(uni_only, uni_reactants)

    for i, smi in enumerate(ZERO_FEASIBLE_BI):
        in_uni = smi in uni_reactants
        feas = mgr_uni_f.feasible_first_reactant_templates(smi, kind="reaction_valid")
        _print(f"   {i + 1}. {smi}")
        _print(f"        in uni test set    : {in_uni}")
        _print(f"        uni R1-feasible    : {len(feas)}  {feas[:5]}")

    _print("\n[2/4] Cross-check against the BI templates pickle structure")
    cfg_bi = load_config("configs/exhausted_search_bi_delta_qed.yaml")
    bi_templates_pkl = load_pickle(resolve_path(cfg_bi["dataset"]["templates_file"]))
    _print(f"   loaded {len(bi_templates_pkl)} bi template entries from "
           f"{cfg_bi['dataset']['templates_file']}")
    _print(f"   container type: {type(bi_templates_pkl).__name__}")

    # Inspect the first few entries to figure out the schema
    sample_entries = list(bi_templates_pkl)[:3]
    _print(f"   first 3 entries (raw): {sample_entries}")
    if isinstance(bi_templates_pkl, dict):
        sample_keys = list(bi_templates_pkl.keys())[:3]
        sample_vals = [bi_templates_pkl[k] for k in sample_keys]
        _print(f"   dict — first 3 keys: {sample_keys}")
        _print(f"   first 3 values type: {[type(v).__name__ for v in sample_vals]}")
        if sample_vals and isinstance(sample_vals[0], dict):
            _print(f"   first value dict keys: {list(sample_vals[0].keys())}")
            _print(f"   first value (truncated): {str(sample_vals[0])[:200]}")

    _print("\n[3/4] Inspect the bi templates pickle as a dict")
    bi_reactants = load_pickle(resolve_path(cfg_bi["dataset"]["test_file"]))
    mgr_bi = ReactionManager(bi_templates_pkl, bi_reactants)
    bi_only_raw = mgr_bi.templates_for_mode("bi")
    _print(f"   templates_for_mode('bi') container type: {type(bi_only_raw).__name__}")
    if isinstance(bi_only_raw, dict):
        bi_only_items = list(bi_only_raw.items())
    else:
        bi_only_items = list(enumerate(bi_only_raw))
    _print(f"   bi-mode template count after filter: {len(bi_only_items)}")
    _print(f"   first 10 template R1 SMARTS heads (sorted by name):")
    summarised = []
    for key, t in bi_only_items:
        if not isinstance(t, dict):
            t = bi_templates_pkl.get(key) if isinstance(bi_templates_pkl, dict) else None
        if not isinstance(t, dict):
            continue
        name = t.get("name") or t.get("Reaction") or f"template_{key}"
        smarts = t.get("smarts") or t.get("reaction_smarts")
        kind = t.get("type", "?")
        # The R1 half of a reaction SMARTS is everything before ">>"
        r1_half = (smarts or "").split(">>")[0] if smarts else "?"
        summarised.append((str(name), kind, r1_half))
    summarised.sort()
    for name, kind, r1 in summarised[:10]:
        _print(f"     type={kind:>3}  {name[:40]:<40s}  R1_pattern={r1[:80]}")
    _print(f"   ... ({len(summarised) - 10} more templates omitted)")

    _print("\n[4/4] Functional-group fingerprint of each zero-feasible molecule")
    _print("       (groups detected via SMARTS substructure search)\n")
    for i, smi in enumerate(ZERO_FEASIBLE_BI):
        _print(f"   {i + 1}. {smi}")
        fg = _functional_groups(smi)
        if not fg:
            _print("        (no recognised functional groups)")
        else:
            for name, n in sorted(fg.items(), key=lambda x: -x[1]):
                _print(f"        - {name}: {n}")
        _print()


if __name__ == "__main__":
    main()
