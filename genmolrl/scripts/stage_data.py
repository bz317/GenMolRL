"""Data staging CLI."""

from __future__ import annotations

import argparse

from genmolrl.chem.datasets import stage_unimolecular_split
from genmolrl.config import repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(repo_root() / "designing-new-molecules/data/train_test_split/uni_molecular"))
    parser.add_argument("--target-dir", default=str(repo_root() / "GenMolRL/data/Uni"))
    args = parser.parse_args()
    staged = stage_unimolecular_split(args.source_dir, args.target_dir)
    for key, value in staged.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
