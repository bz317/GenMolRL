"""Controlled legacy-vs-GenMolRL comparison.

This script intentionally avoids W&B and uses a tiny deterministic data subset.
It compares environment behavior and short seeded PPO/A2C learning runs.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

REPO_ROOT = Path("/home/bz317/new_mol_with_RL_GNN")
GEN_ROOT = REPO_ROOT / "GenMolRL"
LEGACY_ROOT = REPO_ROOT / "other_branches/designing-new-molecules-experiments"
OUT_DIR = GEN_ROOT / "debug" / "controlled_legacy_compare"

sys.path.insert(0, str(LEGACY_ROOT))
sys.path.insert(0, str(GEN_ROOT))

from genmolrl.envs.molecule_design_env import MoleculeDesignEnv as GenEnv  # noqa: E402
from genmolrl.algorithms.a2c.policies import MaskedActorCriticPolicy as GenA2CPolicy  # noqa: E402
from src.models.ppo.envs.molecule_design_env import MoleculeDesignEnv as LegacyEnv  # noqa: E402
from src.models.ppo.train.A2C import MaskedActorCriticPolicy as LegacyA2CPolicy  # noqa: E402


class EpisodeCollector(BaseCallback):
    def __init__(self, eval_factory=None, test_keys=None, checkpoints=None):
        super().__init__()
        self.rewards: list[float] = []
        self.lengths: list[int] = []
        self.step_trace: list[dict] = []
        self.eval_factory = eval_factory
        self.test_keys = list(test_keys or [])
        self.checkpoints = set(checkpoints or [])
        self.checkpoint_evals: list[dict] = []

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", []))
        rewards = np.asarray(self.locals.get("rewards", []))
        dones = np.asarray(self.locals.get("dones", []))
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            action = actions.reshape(-1)[env_idx] if actions.size else None
            reward = rewards.reshape(-1)[env_idx] if rewards.size else None
            done = dones.reshape(-1)[env_idx] if dones.size else None
            self.step_trace.append(
                {
                    "step": int(self.num_timesteps),
                    "env_idx": int(env_idx),
                    "action": None if action is None else int(action),
                    "reward": None if reward is None else float(reward),
                    "done": None if done is None else bool(done),
                    "smiles": info.get("SMILES"),
                    "qed": None if "QED" not in info else float(info["QED"]),
                    "episode_reward": None if info.get("episode") is None else float(info["episode"]["r"]),
                    "episode_length": None if info.get("episode") is None else int(info["episode"]["l"]),
                }
            )
            ep = info.get("episode")
            if ep is not None:
                self.rewards.append(float(ep["r"]))
                self.lengths.append(int(ep["l"]))
        if self.num_timesteps in self.checkpoints and self.eval_factory is not None:
            eval_result = evaluate_model(self.model, self.eval_factory, self.test_keys)
            self.checkpoint_evals.append(
                {
                    "step": int(self.num_timesteps),
                    "mean_reward": eval_result["mean_reward"],
                    "mean_length": eval_result["mean_length"],
                    "rewards": eval_result["rewards"],
                    "lengths": eval_result["lengths"],
                }
            )
        return True


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def subset_data(train_n: int = 128, test_n: int = 32):
    full = load_pickle(GEN_ROOT / "data/Uni/reactants_full.pkl")
    templates = load_pickle(GEN_ROOT / "data/Uni/templates_unimolecolar_explicit.pkl")
    keys = list(full.keys())
    train = {smiles: full[smiles] for smiles in keys[:train_n]}
    test = {smiles: full[smiles] for smiles in keys[train_n : train_n + test_n]}

    gen_train = OUT_DIR / "gen_train.pkl"
    gen_test = OUT_DIR / "gen_test.pkl"
    gen_templates = OUT_DIR / "templates.pkl"
    legacy_train = OUT_DIR / "legacy_train.pkl"
    legacy_test = OUT_DIR / "legacy_test.pkl"
    legacy_templates = OUT_DIR / "legacy_templates.pkl"
    for path, obj in [
        (gen_train, train),
        (gen_test, test),
        (gen_templates, templates),
        (legacy_train, train),
        (legacy_test, test),
        (legacy_templates, templates),
    ]:
        dump_pickle(obj, path)
    return train, test, gen_train, gen_test, gen_templates, legacy_train, legacy_test, legacy_templates


def make_legacy_env(path: Path, templates: Path, seed: int | None = None):
    env = LegacyEnv(
        str(path),
        str(templates),
        max_steps=5,
        use_multidiscrete=False,
        stop_early_penalty=0.0,
        stop_penalty_until_step=3,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_gen_env(path: Path, templates: Path, seed: int | None = None):
    env = GenEnv(
        str(path),
        str(templates),
        reaction_mode="uni",
        algorithm_family="sb3_discrete",
        action_design="discrete",
        masking="reaction_valid",
        reward="delta_qed",
        max_steps=5,
        use_stop_action=True,
        stop_early_penalty=0.0,
        stop_penalty_until_step=3,
        invalid_reaction_penalty=-1.0,
        reward_round_digits=3,
        info_qed_round_digits=3,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def compare_env(train, gen_train, gen_templates, legacy_train, legacy_templates):
    legacy = make_legacy_env(legacy_train, legacy_templates)
    gen = make_gen_env(gen_train, gen_templates)
    counts = {
        "sampled_molecules": 0,
        "compared_actions": 0,
        "reset_start_mismatches": 0,
        "mask_mismatches": 0,
        "product_mismatches": 0,
        "reward_mismatches": 0,
        "termination_mismatches": 0,
        "stop_observation_mismatches": 0,
    }
    examples = []

    for seed in range(20):
        _, legacy_info = legacy.reset(seed=10_000 + seed)
        _, gen_info = gen.reset(seed=10_000 + seed)
        if legacy_info["SMILES"] != gen_info["SMILES"]:
            counts["reset_start_mismatches"] += 1
            examples.append(("reset", seed, legacy_info["SMILES"], gen_info["SMILES"]))

    for smiles in list(train.keys())[:64]:
        counts["sampled_molecules"] += 1
        legacy.current_state = smiles
        legacy.previous_qed = legacy._get_info()["QED"]
        legacy.current_step = 0
        legacy.done = False
        legacy.steps_log = {}
        legacy.permanent_log = {}
        gen.current_state = smiles
        gen.previous_state = smiles
        gen.current_step = 0
        gen.steps_log = {}

        legacy_mask = legacy.action_masks()
        gen_mask = gen.action_masks()
        if not np.array_equal(legacy_mask, gen_mask):
            counts["mask_mismatches"] += 1
            examples.append(("mask", smiles, legacy_mask.tolist(), gen_mask.tolist()))

        valid_actions = [i for i, value in enumerate(legacy_mask[:-1]) if value > 0.5]
        actions = valid_actions[:3] + [len(legacy_mask) - 1]
        for action in actions:
            legacy.current_state = smiles
            legacy.previous_qed = legacy._get_info()["QED"]
            legacy.current_step = 0
            legacy.done = False
            legacy.steps_log = {}
            legacy.permanent_log = {}
            gen.current_state = smiles
            gen.previous_state = smiles
            gen.current_step = 0
            gen.steps_log = {}

            legacy_obs, legacy_reward, legacy_term, legacy_trunc, legacy_info = legacy.step(action)
            gen_obs, gen_reward, gen_term, gen_trunc, gen_info = gen.step(action)
            counts["compared_actions"] += 1
            if action != len(legacy_mask) - 1 and legacy_info.get("SMILES") != gen_info.get("SMILES"):
                counts["product_mismatches"] += 1
                examples.append(("product", smiles, action, legacy_info.get("SMILES"), gen_info.get("SMILES")))
            if float(legacy_reward) != float(gen_reward):
                counts["reward_mismatches"] += 1
                examples.append(("reward", smiles, action, float(legacy_reward), float(gen_reward)))
            if (bool(legacy_term), bool(legacy_trunc)) != (bool(gen_term), bool(gen_trunc)):
                counts["termination_mismatches"] += 1
                examples.append(("terminal", smiles, action, (legacy_term, legacy_trunc), (gen_term, gen_trunc)))
            if action == len(legacy_mask) - 1 and not np.array_equal(legacy_obs, gen_obs):
                counts["stop_observation_mismatches"] += 1
                examples.append(("stop_obs", smiles))
    return {"counts": counts, "examples": examples[:10]}


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def evaluate_model(model, env_factory, test_keys, deterministic: bool = True):
    rewards = []
    lengths = []
    actions_by_start = {}
    for smiles in test_keys:
        env = env_factory()
        env.unwrapped.current_state = smiles
        if hasattr(env.unwrapped, "previous_state"):
            env.unwrapped.previous_state = smiles
        if hasattr(env.unwrapped, "previous_qed"):
            env.unwrapped.previous_qed = env.unwrapped._get_info()["QED"]
        env.unwrapped.current_step = 0
        env.unwrapped.steps_log = {}
        if hasattr(env.unwrapped, "permanent_log"):
            env.unwrapped.permanent_log = {}
        env.unwrapped.initial_qed = env.unwrapped._get_info().get("QED", 0.0)
        obs = env.unwrapped._get_obs()
        done = False
        ep_reward = 0.0
        ep_len = 0
        actions = []
        while not done:
            try:
                action_masks = env.unwrapped.action_masks()
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
            except TypeError:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_len += 1
            actions.append(int(action))
        rewards.append(ep_reward)
        lengths.append(ep_len)
        actions_by_start[smiles] = actions
        env.close()
    return {
        "mean_reward": mean_or_none(rewards),
        "mean_length": mean_or_none(lengths),
        "rewards": rewards,
        "lengths": lengths,
        "actions_by_start": actions_by_start,
    }


def parameter_l2(model_a, model_b):
    total = 0.0
    max_abs = 0.0
    with torch.no_grad():
        for (_, param_a), (_, param_b) in zip(model_a.policy.state_dict().items(), model_b.policy.state_dict().items()):
            diff = (param_a.detach().cpu().float() - param_b.detach().cpu().float()).reshape(-1)
            total += float(torch.sum(diff * diff).item())
            max_abs = max(max_abs, float(torch.max(torch.abs(diff)).item()) if diff.numel() else 0.0)
    return {"l2": float(total**0.5), "max_abs": max_abs}


def compare_step_traces(trace_a: list[dict], trace_b: list[dict]):
    total = min(len(trace_a), len(trace_b))
    mismatches = []
    learning_keys = ["step", "env_idx", "action", "reward", "done", "episode_reward", "episode_length"]
    info_keys = ["smiles", "qed"]
    keys = learning_keys + info_keys
    learning_mismatch_count = 0
    info_mismatch_count = 0
    for idx in range(total):
        row_a = trace_a[idx]
        row_b = trace_b[idx]
        different = {}
        for key in keys:
            if row_a.get(key) != row_b.get(key):
                different[key] = {"legacy": row_a.get(key), "gen": row_b.get(key)}
                if key in learning_keys:
                    learning_mismatch_count += 1
                else:
                    info_mismatch_count += 1
        if different:
            mismatches.append({"trace_index": idx, "differences": different})
            if len(mismatches) >= 10:
                break
    return {
        "legacy_steps": len(trace_a),
        "gen_steps": len(trace_b),
        "compared_steps": total,
        "learning_exact_match": len(trace_a) == len(trace_b) and learning_mismatch_count == 0,
        "all_recorded_fields_exact_match": len(trace_a) == len(trace_b) and not mismatches,
        "learning_mismatch_count": learning_mismatch_count,
        "info_mismatch_count": info_mismatch_count,
        "first_mismatches": mismatches,
    }


def compare_checkpoint_evals(evals_a: list[dict], evals_b: list[dict]):
    total = min(len(evals_a), len(evals_b))
    mismatches = []
    for idx in range(total):
        if evals_a[idx] != evals_b[idx]:
            mismatches.append({"checkpoint_index": idx, "legacy": evals_a[idx], "gen": evals_b[idx]})
            if len(mismatches) >= 5:
                break
    return {
        "legacy_checkpoints": len(evals_a),
        "gen_checkpoints": len(evals_b),
        "compared_checkpoints": total,
        "exact_match": len(evals_a) == len(evals_b) and not mismatches,
        "first_mismatches": mismatches,
    }


def train_pair(algo, gen_train, gen_templates, legacy_train, legacy_templates, test_keys):
    seed = 123
    total_timesteps = 512
    checkpoints = [128, 256, 512]

    def train_one(kind: str):
        set_all_seeds(seed)
        if kind == "legacy":
            eval_factory = lambda: make_legacy_env(legacy_train, legacy_templates)
            env = make_vec_env(
                lambda: make_legacy_env(legacy_train, legacy_templates),
                n_envs=1,
                seed=seed,
            )
        else:
            eval_factory = lambda: make_gen_env(gen_train, gen_templates)
            env = make_vec_env(
                lambda: make_gen_env(gen_train, gen_templates),
                n_envs=1,
                seed=seed,
            )
        if algo == "ppo":
            model = MaskablePPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=2,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.02,
                seed=seed,
                verbose=0,
            )
        elif algo == "a2c":
            model = A2C(
                LegacyA2CPolicy if kind == "legacy" else GenA2CPolicy,
                env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                seed=seed,
                device="cpu",
                verbose=0,
                policy_kwargs={"optimizer_class": RMSpropTFLike, "optimizer_kwargs": {"eps": 1e-5}},
            )
        else:
            raise ValueError(algo)
        callback = EpisodeCollector(eval_factory=eval_factory, test_keys=test_keys, checkpoints=checkpoints)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        env.close()
        return model, callback

    legacy_model, legacy_cb = train_one("legacy")
    gen_model, gen_cb = train_one("gen")

    legacy_eval = evaluate_model(
        legacy_model,
        lambda: make_legacy_env(legacy_train, legacy_templates),
        test_keys,
    )
    gen_eval = evaluate_model(
        gen_model,
        lambda: make_gen_env(gen_train, gen_templates),
        test_keys,
    )
    action_agreement = []
    for smiles in test_keys:
        action_agreement.append(legacy_eval["actions_by_start"][smiles] == gen_eval["actions_by_start"][smiles])

    result = {
        "total_timesteps": total_timesteps,
        "legacy_train_episode_mean_reward": mean_or_none(legacy_cb.rewards),
        "gen_train_episode_mean_reward": mean_or_none(gen_cb.rewards),
        "legacy_train_episode_mean_length": mean_or_none(legacy_cb.lengths),
        "gen_train_episode_mean_length": mean_or_none(gen_cb.lengths),
        "training_step_trace_comparison": compare_step_traces(legacy_cb.step_trace, gen_cb.step_trace),
        "checkpoint_eval_comparison": compare_checkpoint_evals(legacy_cb.checkpoint_evals, gen_cb.checkpoint_evals),
        "checkpoint_evals": {
            "legacy": legacy_cb.checkpoint_evals,
            "gen": gen_cb.checkpoint_evals,
        },
        "parameter_diff": parameter_l2(legacy_model, gen_model),
        "legacy_eval": {k: v for k, v in legacy_eval.items() if k != "actions_by_start"},
        "gen_eval": {k: v for k, v in gen_eval.items() if k != "actions_by_start"},
        "eval_action_sequence_agreement": {
            "count": int(sum(action_agreement)),
            "total": len(action_agreement),
            "fraction": float(np.mean(action_agreement)) if action_agreement else None,
        },
    }
    return result


def main():
    set_all_seeds(123)
    train, test, gen_train, gen_test, gen_templates, legacy_train, legacy_test, legacy_templates = subset_data()
    env_result = compare_env(train, gen_train, gen_templates, legacy_train, legacy_templates)
    test_keys = list(test.keys())[:16]
    results = {
        "data": {"train_n": len(train), "test_n": len(test), "eval_n": len(test_keys)},
        "environment": env_result,
        "ppo": train_pair("ppo", gen_train, gen_templates, legacy_train, legacy_templates, test_keys),
        "a2c": train_pair("a2c", gen_train, gen_templates, legacy_train, legacy_templates, test_keys),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "comparison_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
