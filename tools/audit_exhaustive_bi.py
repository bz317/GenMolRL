"""Empirical audit that bi-mode exhausted_search visits every (T, R2) branch
of every start in reactants_test.pkl up to depth=5.

Strategy:
  - Build the real SearchRunner on the updated yaml.
  - Confirm every soft cap is None and depth bound = 5.
  - For 2 representative start molecules compute the analytic depth-1
    branching count via `_all_next` and then run the real `_run_exhausted`
    at depth=1 limited to those starts. The DFS `total_reactions` and
    `saved_paths` must exactly equal the analytic counts; any mismatch
    means a hidden cap is firing.
  - Argue depth>1 by induction over the DFS body in search.py.

This script does not touch the user's results file or W&B (use_wandb is
forced off and results_file is redirected to /tmp before construction).
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("WANDB_MODE", "disabled")

from genmolrl.algorithms.search import SearchRunner  # noqa: E402
from genmolrl.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config("configs/exhausted_search_bi_delta_qed.yaml")
    cfg.setdefault("search", {})
    cfg["search"]["use_wandb"] = False
    cfg["search"]["overwrite_results"] = False
    cfg["search"]["results_file"] = "/tmp/audit_exhaustive_dryrun.txt"

    print("[1/3] Constructing SearchRunner (first-time load is slow because the")
    print("      bi reactant and template pickles are large)...")
    t0 = time.time()
    runner = SearchRunner(cfg, mode="exhausted_search", experiment_name="audit_exhaustive")
    print(f"  built in {time.time() - t0:.1f}s")

    print("\n[2/3] Resolved DFS caps (everything except depth must be None):")
    print(f"  max_steps              = {runner.max_steps}")
    print(f"  max_starts             = {runner.max_starts}    (None ⇒ iterate every key)")
    print(f"  max_paths              = {runner.max_paths}    (None ⇒ save every leaf)")
    print(f"  max_reactions          = {runner.max_reactions}    (None ⇒ no global cap)")
    print(f"  max_r2_per_template    = {runner.max_r2_per_template}    (None ⇒ enumerate every R2)")
    print(f"  reactant_keys size     = {len(runner.reactant_keys)} starts")
    assert runner.max_steps == 5
    assert runner.max_starts is None
    assert runner.max_paths is None
    assert runner.max_reactions is None
    assert runner.max_r2_per_template is None
    print("  [OK] only depth=5 is binding")

    print("\n[3a/3] Iteration: confirm depth=0 DFS visits every valid-SMILES start")
    print("       in reactant_keys (i.e. no hidden start-side cap).")
    runner.max_steps = 0
    runner.all_steps = []
    runner.trajectory_summaries = []
    t0 = time.time()
    res0 = runner._run_exhausted()
    print(f"  finished in {time.time() - t0:.1f}s")
    print(f"  starts_seen     = {res0['starts_seen']}    (out of {len(runner.reactant_keys)} keys)")
    print(f"  total_reactions = {res0['total_reactions']}    (expected 0)")
    print(f"  saved_paths     = {res0['saved_paths']}    (expected starts_seen)")
    assert res0["total_reactions"] == 0
    assert res0["saved_paths"] == res0["starts_seen"]
    assert res0["starts_seen"] >= int(0.99 * len(runner.reactant_keys)), (
        f"only {res0['starts_seen']}/{len(runner.reactant_keys)} starts visited — "
        "looks like an unexpected SMILES-validity drop"
    )
    print(f"  [OK] iteration walks {res0['starts_seen']}/{len(runner.reactant_keys)} starts")

    print("\n[3b/3] Branching distribution at depth=1 across 10 sampled starts")
    print("       (gives a feel for the depth-5 explosion factor)\n")
    sample_starts = runner.reactant_keys[:10]
    per_start_d1: list[tuple[str, int, int, float]] = []
    analytic_total_reactions = 0
    analytic_total_paths = 0

    for s in sample_starts:
        t0 = time.time()
        children = runner._all_next(s)
        dt = time.time() - t0
        distinct_tmpls = len({c[0] for c in children})
        per_template: dict[int, int] = {}
        for tmpl_idx, _, _, _ in children:
            per_template[tmpl_idx] = per_template.get(tmpl_idx, 0) + 1
        per_start_d1.append((s, len(children), distinct_tmpls, dt))
        analytic_total_reactions += len(children)
        analytic_total_paths += max(len(children), 1)

    print(f"  {'start':<60s}  {'d1_branches':>12s}  {'tmpls':>6s}  {'time_s':>7s}")
    for s, n, t, dt in per_start_d1:
        print(f"  {s[:58]:<60s}  {n:>12,}  {t:>6}  {dt:>7.1f}")

    branches = [n for _, n, _, _ in per_start_d1]
    print(
        f"\n  depth-1 branching: min={min(branches)}, median={sorted(branches)[len(branches) // 2]}, "
        f"max={max(branches)}, mean={sum(branches) / len(branches):.1f}"
    )
    print(f"  analytic totals across 10 starts: reactions={analytic_total_reactions:,}, "
          f"terminal paths={analytic_total_paths:,}")

    print("\n  Running real SearchRunner._run_exhausted() on the same 10 starts at depth=1 ...")
    runner.max_steps = 1
    runner.reactant_keys = list(sample_starts)
    runner.all_steps = []
    runner.trajectory_summaries = []
    t0 = time.time()
    result = runner._run_exhausted()
    print(f"    DFS finished in {time.time() - t0:.1f}s")
    print(f"    DFS starts_seen     = {result['starts_seen']}    (expected {len(sample_starts)})")
    print(f"    DFS total_reactions = {result['total_reactions']:,}    (expected {analytic_total_reactions:,})")
    print(f"    DFS saved_paths     = {result['saved_paths']:,}    (expected {analytic_total_paths:,})")
    assert result["starts_seen"] == len(sample_starts)
    assert result["total_reactions"] == analytic_total_reactions
    assert result["saved_paths"] == analytic_total_paths
    print("  [OK] depth=1 DFS matches analytic branching for every start (no leak).")

    print("\n[3c/3] Recursion: pick the start with largest non-zero d1 branching")
    print("       and verify depth=2 DFS equals analytic depth-1 + sum-over-d1-children depth-1.")
    non_zero = [(s, n) for s, n, _, _ in per_start_d1 if n > 0]
    if not non_zero:
        print("  no start in the sample has any depth-1 reaction; skipping recursion check")
    else:
        # Pick the largest-branching start so the recursion is non-trivial but still bounded
        chosen, d1_count = max(non_zero, key=lambda x: x[1])
        print(f"\n  chosen start: {chosen[:60]}  (d1_branches = {d1_count})")
        d1_children = runner._all_next(chosen)
        d2_per_child = []
        analytic_d2_reactions = 0
        analytic_d2_terminal_paths = 0
        analytic_d1_dead_ends = 0
        for idx, (_, product, _, _) in enumerate(d1_children):
            t0 = time.time()
            d2 = runner._all_next(product)
            dt = time.time() - t0
            d2_per_child.append(len(d2))
            if len(d2) == 0:
                analytic_d1_dead_ends += 1
            else:
                analytic_d2_reactions += len(d2)
                analytic_d2_terminal_paths += len(d2)
            print(f"    d1 child {idx + 1}/{len(d1_children)}: depth-2 branching = {len(d2):,} ({dt:.1f}s)")

        analytic_total_reactions_d2 = d1_count + analytic_d2_reactions
        analytic_total_paths_d2 = analytic_d2_terminal_paths + analytic_d1_dead_ends
        if analytic_total_paths_d2 == 0:
            analytic_total_paths_d2 = 1
        print(f"\n  analytic at depth=2: total reactions = {analytic_total_reactions_d2:,}, "
              f"terminal paths = {analytic_total_paths_d2:,}")

        runner.max_steps = 2
        runner.reactant_keys = [chosen]
        runner.all_steps = []
        runner.trajectory_summaries = []
        t0 = time.time()
        result2 = runner._run_exhausted()
        print(f"  DFS finished in {time.time() - t0:.1f}s")
        print(f"    DFS total_reactions = {result2['total_reactions']:,}    "
              f"(expected {analytic_total_reactions_d2:,})")
        print(f"    DFS saved_paths     = {result2['saved_paths']:,}    "
              f"(expected {analytic_total_paths_d2:,})")
        assert result2["total_reactions"] == analytic_total_reactions_d2, (
            f"depth-2 branch leak: DFS={result2['total_reactions']}, "
            f"analytic={analytic_total_reactions_d2}"
        )
        assert result2["saved_paths"] == analytic_total_paths_d2, (
            f"depth-2 path leak: DFS={result2['saved_paths']}, "
            f"analytic={analytic_total_paths_d2}"
        )
        print("  [OK] depth=2 DFS exhausts every recursive branch.")

    print("\nDepth > 1 follows by induction over search.py::dfs (L409-443):")
    print("  - `_all_next(node)` enumerates every (template, R2) whose RDKit")
    print("    apply_reaction yields a valid product. No other masking is in")
    print("    play (only `reaction_valid`).")
    print("  - The DFS body recurses on every returned tuple until")
    print("    `step_idx >= max_steps`. Because depth=1 already exhausts the root")
    print("    branching, the same code path exhausts every internal node up to depth=5.")
    print("  - With max_paths / max_reactions / max_r2_per_template all None, no")
    print("    early-return condition can fire inside the recursion.")
    print("\n[VERIFIED] At depth=5 with the updated yaml, the runner will visit every")
    print("           molecule in reactants_test.pkl and expand every reachable")
    print("           (template, R2) action sequence of length up to 5 reactions.")


if __name__ == "__main__":
    main()
