#!/usr/bin/env python3
"""Download the *exact* training config W&B stored for a run.

GenMolRL calls ``wandb.init(..., config=config)`` with the fully merged dict
after YAML load + CLI overrides (see ``init_wandb`` in
``genmolrl/algorithms/common.py``). That snapshot is what ``run.config`` returns
from the API — it does **not** change when you edit local YAML later.

Examples::

    cd /path/to/GenMolRL/GenMolRL
    python tools/fetch_wandb_run_config.py \\
        --run-path boqiaoz-cambridge/GenMolRL_Bi/runs/pr9mkz9b \\
        --out /tmp/pr9mkz9b_config.yaml

    python tools/fetch_wandb_run_config.py \\
        --entity boqiaoz-cambridge --project GenMolRL_Bi --id pr9mkz9b \\
        --format json

Auth (first match wins):

1. ``WANDB_API_KEY`` environment variable
2. ``WANDB_API_KEY_FILE`` — path to a file containing only the key
3. ``<repo-outer>/wandb_api_key.txt`` — one directory above ``GenMolRL/``
   (the usual layout: ``.../GenMolRL/wandb_api_key.txt`` next to
   ``.../GenMolRL/GenMolRL/``). HPC SLURM scripts only activate conda; they
   do not export W&B — use one of the above or ``wandb login``.

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import wandb


def _ensure_wandb_api_key() -> None:
    """Populate ``WANDB_API_KEY`` from env or repo-local file if missing."""
    if os.environ.get("WANDB_API_KEY", "").strip():
        return
    paths: list[Path] = []
    key_file = os.environ.get("WANDB_API_KEY_FILE", "").strip()
    if key_file:
        paths.append(Path(key_file).expanduser())
    # .../GenMolRL/GenMolRL/tools/<this>.py -> .../GenMolRL/wandb_api_key.txt
    outer = Path(__file__).resolve().parents[2]
    paths.append(outer / "wandb_api_key.txt")
    cwd = Path.cwd()
    paths.append(cwd / "wandb_api_key.txt")
    paths.append(cwd.parent / "wandb_api_key.txt")

    for p in paths:
        try:
            if p.is_file():
                key = p.read_text(encoding="utf-8").strip()
                if key:
                    os.environ["WANDB_API_KEY"] = key
                    return
        except OSError:
            continue


def _config_to_plain(run) -> dict[str, Any]:
    """Turn ``wandb``'s config mapping into a JSON/YAML-friendly nested dict."""
    cfg = run.config
    # Public API: iterate keys and resolve values (handles nested + wandb Value).
    out: dict[str, Any] = {}
    for key in cfg.keys():
        out[key] = cfg[key]
    return out


def _strip_internal_keys(tree: Any) -> Any:
    """Remove W&B-internal keys like ``_wandb`` from nested dicts."""
    if isinstance(tree, dict):
        return {
            k: _strip_internal_keys(v)
            for k, v in tree.items()
            if not str(k).startswith("_wandb")
        }
    if isinstance(tree, list):
        return [_strip_internal_keys(v) for v in tree]
    return tree


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--run-path",
        help="Full path, e.g. boqiaoz-cambridge/GenMolRL_Bi/runs/pr9mkz9b",
    )
    g.add_argument("--entity", help="With --project and --id")
    p.add_argument("--project", help="With --entity and --id")
    p.add_argument("--id", metavar="RUN_ID", help="With --entity and --project")
    p.add_argument(
        "--out",
        type=Path,
        help="Write YAML here (default: stdout)",
    )
    p.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format (default: yaml)",
    )
    p.add_argument(
        "--keep-wandb-keys",
        action="store_true",
        help="Keep internal ``_wandb`` subtree if present",
    )
    args = p.parse_args()

    if args.run_path:
        path = args.run_path.strip().strip("/")
        parts = path.split("/")
        # Accept ``entity/project/runs/<id>`` (W&B URL style) or ``entity/project/<id>``.
        if len(parts) == 4 and parts[2] == "runs":
            entity, project, _, run_id = parts[0], parts[1], parts[2], parts[3]
        elif len(parts) == 3:
            entity, project, run_id = parts[0], parts[1], parts[2]
        else:
            p.error(
                "--run-path must look like entity/project/runs/<id> or entity/project/<id>, "
                f"got {args.run_path!r}"
            )
    else:
        if not (args.entity and args.project and args.id):
            p.error("--entity, --project, and --id are required together")
        entity, project, run_id = args.entity, args.project, args.id

    _ensure_wandb_api_key()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/runs/{run_id}")

    plain = _config_to_plain(run)
    if not args.keep_wandb_keys:
        plain = _strip_internal_keys(plain)

    header = (
        f"# W&B run config snapshot\n"
        f"# run: {entity}/{project}/{run_id}\n"
        f"# name: {run.name}\n"
        f"# state: {run.state}\n"
        f"# created: {run.created_at}\n"
    )

    if args.format == "json":
        text = header + json.dumps(plain, indent=2, default=str)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            print("PyYAML is required for --format yaml: pip install pyyaml", file=sys.stderr)
            raise SystemExit(1) from e
        text = header + yaml.safe_dump(plain, sort_keys=False, default_flow_style=False)

    if args.out:
        args.out.write_text(text)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
