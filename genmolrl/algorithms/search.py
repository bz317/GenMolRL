"""Random and greedy search baselines over the shared GenMolRL chemistry layer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

import wandb
from genmolrl.chem.datasets import load_pickle
from genmolrl.chem.reaction_manager import BI_TYPE, ReactionManager
from genmolrl.config import repo_root
from genmolrl.envs.rewards import RewardFunction, qed


@dataclass
class SearchStep:
    path_id: int
    step: int
    reactant: str
    template: str
    product: str
    qed: float
    reward: float
    second_reactant: str | None = None


def _valid_smiles(smiles: str | None) -> bool:
    return bool(smiles) and Chem.MolFromSmiles(smiles) is not None


class SearchRunner:
    """Non-neural molecule search baseline.

    `random_search` randomly chooses one valid template and, for Bi templates, one
    valid R2. `greedy_search` enumerates valid candidates at each step and picks
    the candidate with the highest configured reward.
    """

    def __init__(self, config: dict, experiment_name: str, mode: str):
        if mode not in {"random_search", "greedy_search"}:
            raise ValueError(f"Unsupported search mode: {mode}")
        self.config = config
        self.mode = mode
        self.experiment_name = experiment_name
        self.dataset = config["dataset"]
        self.search_cfg = config.get("search", {})
        self.masking = config["masking"]
        self.reward_fn = RewardFunction(
            config["reward"],
            invalid_penalty=float(config.get("env", {}).get("invalid_reaction_penalty", -1.0)),
            round_digits=config.get("env", {}).get("reward_round_digits"),
        )
        self.max_steps = int(self.search_cfg.get("max_steps", config.get("env", {}).get("max_steps", 5)))
        self.max_paths = int(self.search_cfg.get("max_paths", 100))
        self.max_attempts = int(self.search_cfg.get("max_attempts", 1000))
        self.max_reactions = int(self.search_cfg.get("max_reactions", 10000))
        self.max_r2_per_template = int(self.search_cfg.get("max_r2_per_template", 100))
        self.overwrite_results = bool(self.search_cfg.get("overwrite_results", True))
        self.seed = int(config.get("training", {}).get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)

        reactants = load_pickle(repo_root() / self.dataset["training_file"])
        templates = load_pickle(repo_root() / self.dataset["templates_file"])
        all_manager = ReactionManager(templates, reactants)
        templates_for_mode = all_manager.templates_for_mode(config["reaction_mode"])
        self.manager = ReactionManager(templates_for_mode, reactants)
        self.reactant_keys = list(reactants.keys())
        self.result_file = Path(
            self.search_cfg.get(
                "results_file",
                f"GenMolRL/runs/{self.mode}_results.txt",
            )
        )
        if not self.result_file.is_absolute():
            self.result_file = repo_root() / self.result_file
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        if self.overwrite_results and self.result_file.exists():
            self.result_file.unlink()

        wandb_disabled = str(self.search_cfg.get("use_wandb", True)).lower() in {"0", "false", "no", "off"}
        self.use_wandb = not wandb_disabled
        if self.use_wandb:
            init_kw = {
                "project": config.get("project", "MolSynthRL"),
                "name": experiment_name,
                "job_type": self.mode,
                "config": config,
                "save_code": True,
                "resume": "never",
            }
            if config.get("entity"):
                init_kw["entity"] = config["entity"]
            wandb.init(**init_kw)
        self.all_steps: list[SearchStep] = []

    def _template_name(self, idx: int) -> str:
        return str(self.manager.templates[idx].get("name", self.manager.templates[idx].get("Reaction", idx)))

    def _candidate_products(self, current: str, template_idx: int) -> list[tuple[str, str | None, float]]:
        template = self.manager.templates[template_idx]
        if template.get("type") == BI_TYPE:
            partners = self.manager.get_valid_reactants(template_idx)
            if self.mode == "random_search":
                random.shuffle(partners)
                partners = partners[:1]
            else:
                partners = partners[: self.max_r2_per_template]
            out = []
            for r2 in partners:
                product = self.manager.apply_reaction(current, template, r2)
                if product:
                    out.append((product, r2, self.reward_fn.step_reward(current, product)))
            return out
        product = self.manager.apply_reaction(current, template, None)
        if not product:
            return []
        return [(product, None, self.reward_fn.step_reward(current, product))]

    def _valid_template_indices(self, current: str) -> list[int]:
        return self.manager.feasible_first_reactant_templates(current, kind=self.masking)

    def _sample_start(self) -> str:
        for _ in range(100):
            candidate = random.choice(self.reactant_keys)
            if _valid_smiles(candidate):
                return candidate
        raise RuntimeError("Could not sample a valid starting molecule.")

    def _choose_next(self, current: str) -> tuple[int, str, str | None, float] | None:
        candidates = []
        template_indices = self._valid_template_indices(current)
        if self.mode == "random_search":
            random.shuffle(template_indices)
        for idx in template_indices:
            products = self._candidate_products(current, idx)
            if self.mode == "random_search" and products:
                product, r2, reward = random.choice(products)
                return idx, product, r2, reward
            for product, r2, reward in products:
                candidates.append((reward, idx, product, r2))
        if not candidates:
            return None
        reward, idx, product, r2 = max(candidates, key=lambda x: x[0])
        return idx, product, r2, reward

    def _write_report(self, summary: dict) -> None:
        fields = [
            "path_id",
            "step",
            "reactant",
            "template",
            "product",
            "qed",
            "reward",
            "second_reactant",
        ]
        with self.result_file.open("w", encoding="utf-8") as f:
            f.write("# GenMolRL search results\n\n")
            f.write("[summary]\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n[steps]\n")
            f.write("\t".join(fields) + "\n")
            for step in self.all_steps:
                row = step.__dict__
                f.write("\t".join("" if row[k] is None else str(row[k]) for k in fields) + "\n")

    def run(self):
        saved_paths = 0
        attempts = 0
        total_reactions = 0
        best_qed = 0.0
        while (
            saved_paths < self.max_paths
            and attempts < self.max_attempts
            and total_reactions < self.max_reactions
        ):
            attempts += 1
            current = self._sample_start()
            initial_qed = qed(current)
            path_rows: list[SearchStep] = [
                SearchStep(
                    path_id=saved_paths,
                    step=0,
                    reactant=current,
                    template="START",
                    product=current,
                    qed=round(initial_qed, 3),
                    reward=0.0,
                    second_reactant=None,
                )
            ]
            for step_idx in range(1, self.max_steps + 1):
                if total_reactions >= self.max_reactions:
                    break
                chosen = self._choose_next(current)
                if chosen is None:
                    break
                template_idx, product, r2, reward = chosen
                q = qed(product)
                best_qed = max(best_qed, q)
                path_rows.append(
                    SearchStep(
                        path_id=saved_paths,
                        step=step_idx,
                        reactant=current,
                        template=self._template_name(template_idx),
                        product=product,
                        qed=round(q, 3),
                        reward=reward,
                        second_reactant=r2,
                    )
                )
                current = product
                total_reactions += 1
            reaction_steps = len(path_rows) - 1
            if reaction_steps > 0:
                self.all_steps.extend(path_rows)
                saved_paths += 1
            if self.use_wandb:
                terminal_qed = path_rows[-1].qed
                wandb.log(
                    {
                        "train/global_step": attempts,
                        "search/attempts": attempts,
                        "search/saved_paths": saved_paths,
                        "search/total_reactions": total_reactions,
                        "search/last_path_length": reaction_steps,
                        "search/last_initial_qed": round(initial_qed, 3),
                        "search/last_terminal_qed": terminal_qed,
                        "search/last_net_qed_gain": terminal_qed - round(initial_qed, 3),
                        "search/best_qed": best_qed,
                    },
                    step=attempts,
                )
        summary = {
            "attempts": attempts,
            "saved_paths": saved_paths,
            "total_reactions": total_reactions,
            "best_qed": best_qed,
            "results_file": str(self.result_file),
        }
        self._write_report(summary)
        if self.use_wandb:
            wandb.finish()
        return summary


def train(config: dict, experiment_name: str, *, mode: str):
    return SearchRunner(config, experiment_name, mode).run()
