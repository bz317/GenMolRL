"""Data staging CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from genmolrl.chem.datasets import load_pickle
from genmolrl.chem.datasets import stage_unimolecular_split
from genmolrl.config import project_root


def _existing_unimolecular_layout(target_dir: str | Path) -> dict[str, str | int]:
    target = Path(target_dir)
    train_p = target / "reactants_train.pkl"
    test_p = target / "reactants_test.pkl"
    template_p = target / "templates_unimolecolar_explicit.pkl"
    starts_p = target / "eval_start_smiles.txt"
    missing = [str(p) for p in [train_p, test_p, template_p] if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "GenMolRL Uni data is not staged. Missing: "
            + ", ".join(missing)
            + ". Pass --source-dir explicitly if you want to stage from an external split."
        )
    return {
        "training_file": str(train_p),
        "test_file": str(test_p),
        "templates_file": str(template_p),
        "eval_start_smiles_file": str(starts_p) if starts_p.is_file() else "",
        "n_test_molecules": len(load_pickle(test_p)),
    }


def _display_value(value):
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", help="Optional external uni split directory to copy from")
    parser.add_argument("--target-dir", default=str(project_root() / "data/Uni"))
    args = parser.parse_args()
    staged = (
        stage_unimolecular_split(args.source_dir, args.target_dir)
        if args.source_dir
        else _existing_unimolecular_layout(args.target_dir)
    )
    for key, value in staged.items():
        print(f"{key}={_display_value(value)}")


if __name__ == "__main__":
    main()
