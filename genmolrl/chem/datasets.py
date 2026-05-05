"""Dataset loading and staging helpers."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def stage_unimolecular_split(source_dir: str | Path, target_dir: str | Path) -> dict[str, str | int]:
    """Stage the canonical GenMolRL Uni dataset layout.

    The raw split files keep the same names as the source split:
    `reactants_train.pkl`, `reactants_val.pkl`, and
    `templates_unimolecolar_explicit.pkl`. We also create compatibility
    derivatives used by the current PPO/A2C/TD3 launchers:
    `reactants_full.pkl` and `eval_start_smiles.txt`.
    """
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    train_p = target / "reactants_train.pkl"
    val_p = target / "reactants_val.pkl"
    template_p = target / "templates_unimolecolar_explicit.pkl"
    full_p = target / "reactants_full.pkl"
    starts_p = target / "eval_start_smiles.txt"

    shutil.copy2(source / "reactants_train.pkl", train_p)
    shutil.copy2(source / "reactants_val.pkl", val_p)
    shutil.copy2(source / "templates_unimolecolar_explicit.pkl", template_p)

    train = load_pickle(train_p)
    val = load_pickle(val_p)
    full = dict(train)
    full.update(val)
    dump_pickle(full, full_p)

    with starts_p.open("w", encoding="utf-8") as f:
        for smiles in val:
            f.write(f"{smiles}\n")

    return {
        "training_file": str(full_p),
        "validation_file": str(full_p),
        "templates_file": str(template_p),
        "eval_start_smiles_file": str(starts_p),
        "n_eval_episodes": len(val),
    }
