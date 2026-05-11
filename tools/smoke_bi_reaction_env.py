"""Smoke test: confirm Bi-PPO env can actually execute bi-reactions end-to-end.

Mirrors the runtime configuration of `configs/ppo_bi_masked_delta_qed.yaml`:
  - reaction_mode = "bi"
  - algorithm_family = "sb3_multidiscrete"
  - masking = "reaction_valid"   (the previously buggy mask we just fixed)

For a handful of starting molecules drawn from data/Bi/reactants_test.pkl it:
  1. resets the env, computes the action mask, and checks that >=1 BI template
     is marked valid in the template slice of the mask;
  2. picks the first valid bi-template T, finds an R2 in the reactant pool
     whose pattern matches T's second reagent, takes the env step, and
     verifies the resulting product is (a) non-empty, (b) different from the
     start, (c) consistent with a direct `apply_reaction(R1, T, R2)` call;
  3. also exercises the unmasked-R2 failure path (random pool R2 vs T) to
     confirm the env returns the configured invalid penalty + truncates the
     episode rather than silently producing a uni product.

Run with:
    cd GenMolRL && WANDB_MODE=disabled PYTHONPATH=. python tools/smoke_bi_reaction_env.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from genmolrl.chem.reaction_manager import BI_TYPE
from genmolrl.envs.molecule_design_env import MoleculeDesignEnv


def main() -> None:
    env = MoleculeDesignEnv(
        reactant_file="data/Bi/reactants_train.pkl",
        template_file="data/Bi/templates.pkl",
        reaction_mode="bi",
        algorithm_family="sb3_multidiscrete",
        action_design="discrete",
        masking="reaction_valid",
        reward="delta_qed",
        max_steps=5,
        start_strategy="random_pool",
        start_pool_file="data/Bi/reactants_test.pkl",
        use_stop_action=True,
        invalid_reaction_penalty=-1.0,
        reward_round_digits=3,
        info_qed_round_digits=3,
        append_action_mask_to_obs=False,
    )
    num_templates = env.num_templates
    num_reactants = len(env.reactants)
    print(f"env action_space={env.action_space} num_templates={num_templates} num_reactants={num_reactants}")

    n_starts = 8
    bi_success = 0
    bi_seen_any_valid = 0
    invalid_path_ok = 0
    rng = random.Random(123)

    for trial in range(n_starts):
        obs, info = env.reset(seed=trial)
        start = info["SMILES"]
        mask = env.action_masks()
        template_mask_with_stop = mask[: num_templates + 1]
        template_mask = template_mask_with_stop[:num_templates]
        valid_total = int(template_mask.sum())

        valid_bi = [
            idx for idx in np.where(template_mask > 0)[0]
            if env.templates[int(idx)].get("type") == BI_TYPE
        ]
        valid_uni = [
            idx for idx in np.where(template_mask > 0)[0]
            if env.templates[int(idx)].get("type") != BI_TYPE
        ]

        print(
            f"\n[trial {trial}] start={start[:60]!r:<62} "
            f"valid_total={valid_total} valid_bi={len(valid_bi)} valid_uni={len(valid_uni)}"
        )

        if not valid_bi:
            print(f"  no bi-template valid for this start; skipping productive step")
            continue
        bi_seen_any_valid += 1

        t_idx = int(valid_bi[0])
        template = env.templates[t_idx]
        partners = env.reaction_manager.get_valid_reactants(t_idx)
        partner_product = None
        partner_r2 = None
        for r2 in partners:
            prod = env.reaction_manager.apply_reaction(start, template, r2)
            if prod:
                partner_r2 = r2
                partner_product = prod
                break
        if partner_r2 is None:
            print(f"  WARN: template marked valid but no R2 in pool yields a product; mask/pool mismatch")
            continue
        r2_idx = env.reactant_keys.index(partner_r2)

        action = np.array([t_idx, r2_idx], dtype=np.int64)
        obs, reward, terminated, truncated, step_info = env.step(action)
        new_smiles = step_info["SMILES"]
        reaction_failed = bool(step_info.get("reaction_failed", False))
        print(
            f"  bi-template[{t_idx}]={template.get('name', '?')!r} "
            f"R2={partner_r2!r}  reward={reward:.3f} "
            f"failed={reaction_failed} terminated={terminated} truncated={truncated}"
        )
        print(f"  product       : {new_smiles}")
        print(f"  expected (RDKit): {partner_product}")
        assert not reaction_failed, "env reported reaction_failed for a known-valid bi step"
        assert new_smiles == partner_product, (
            f"env product diverges from direct apply_reaction:\n"
            f"  env     : {new_smiles}\n"
            f"  expected: {partner_product}"
        )
        assert new_smiles != start, "bi product equals start; R2 was likely dropped"
        bi_success += 1

        env.reset(seed=trial)
        bad_r2 = rng.choice(env.reactant_keys)
        action = np.array([t_idx, env.reactant_keys.index(bad_r2)], dtype=np.int64)
        obs, reward, terminated, truncated, step_info = env.step(action)
        if step_info.get("reaction_failed"):
            print(
                f"  [unmasked-R2 path] random R2={bad_r2[:40]!r} → reaction_failed, "
                f"reward={reward:.3f} truncated={truncated} (expected: invalid penalty)"
            )
            assert truncated and reward == -1.0
            invalid_path_ok += 1
        else:
            print(
                f"  [unmasked-R2 path] random R2={bad_r2[:40]!r} happened to react: "
                f"reward={reward:.3f}; product={step_info['SMILES'][:60]}"
            )

    print(
        f"\nSUMMARY: starts={n_starts} bi_valid_in_mask={bi_seen_any_valid} "
        f"bi_step_succeeded={bi_success} unmasked_R2_failure_path_verified={invalid_path_ok}"
    )
    assert bi_seen_any_valid > 0, "no start produced any valid bi-template — mask still broken?"
    assert bi_success > 0, "no bi-template step actually executed — env wiring broken"


if __name__ == "__main__":
    main()
