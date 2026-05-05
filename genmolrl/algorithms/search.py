"""Non-neural search baselines over the shared GenMolRL chemistry layer."""

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
    the candidate with the highest configured reward. `exhausted_search`
    enumerates every valid trajectory from each configured test molecule.
    """

    def __init__(self, config: dict, experiment_name: str, mode: str):
        if mode not in {"random_search", "greedy_search", "exhausted_search"}:
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
        env_cfg = config.get("env", {})
        self.max_steps = int(
            config.get(
                "max_episode_len",
                self.search_cfg.get(
                    "max_episode_len",
                    self.search_cfg.get("max_steps", env_cfg.get("max_episode_len", env_cfg.get("max_steps", 5))),
                ),
            )
        )
        self.max_paths = self._optional_int("max_paths", None if mode == "exhausted_search" else 100)
        self.max_attempts = self._optional_int("max_attempts", 1000)
        self.max_reactions = self._optional_int("max_reactions", None if mode == "exhausted_search" else 10000)
        self.max_starts = self._optional_int("max_starts", None)
        self.max_r2_per_template = self._optional_int("max_r2_per_template", None if mode == "exhausted_search" else 100)
        self.overwrite_results = bool(self.search_cfg.get("overwrite_results", True))
        self.seed = int(config.get("training", {}).get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)

        reactant_pool_file = self.dataset.get("test_file")
        if reactant_pool_file is None:
            raise KeyError("dataset.test_file must be set for search baselines")
        reactants = load_pickle(repo_root() / reactant_pool_file)
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

    def _optional_int(self, key: str, default: int | None) -> int | None:
        value = self.search_cfg.get(key, default)
        if value is None:
            return None
        return int(value)

    def _template_name(self, idx: int) -> str:
        return str(self.manager.templates[idx].get("name", self.manager.templates[idx].get("Reaction", idx)))

    def _candidate_products(self, current: str, template_idx: int) -> list[tuple[str, str | None, float]]:
        template = self.manager.templates[template_idx]
        if template.get("type") == BI_TYPE:
            partners = self.manager.get_valid_reactants(template_idx)
            if self.mode == "random_search":
                random.shuffle(partners)
                partners = partners[:1]
            elif self.max_r2_per_template is not None:
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

    def _all_next(self, current: str) -> list[tuple[int, str, str | None, float]]:
        candidates = []
        for idx in self._valid_template_indices(current):
            for product, r2, reward in self._candidate_products(current, idx):
                candidates.append((idx, product, r2, reward))
        return candidates

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

    def _reached_limit(self, saved_paths: int, total_reactions: int) -> bool:
        if self.max_paths is not None and saved_paths >= self.max_paths:
            return True
        return self.max_reactions is not None and total_reactions >= self.max_reactions

    def _log_search_progress(
        self,
        *,
        attempts: int,
        saved_paths: int,
        total_reactions: int,
        last_path_length: int,
        initial_qed: float,
        terminal_qed: float,
        best_qed: float,
    ) -> None:
        if not self.use_wandb:
            return
        wandb.log(
            {
                "train/global_step": attempts,
                "search/attempts": attempts,
                "search/saved_paths": saved_paths,
                "search/total_reactions": total_reactions,
                "search/last_path_length": last_path_length,
                "search/last_initial_qed": round(initial_qed, 3),
                "search/last_terminal_qed": terminal_qed,
                "search/last_net_qed_gain": terminal_qed - round(initial_qed, 3),
                "search/best_qed": best_qed,
            },
            step=attempts,
        )

    def _summary(self, attempts: int, saved_paths: int, total_reactions: int, best_qed: float) -> dict:
        return {
            "attempts": attempts,
            "saved_paths": saved_paths,
            "total_reactions": total_reactions,
            "best_qed": best_qed,
            "results_file": str(self.result_file),
        }

    def _finish(self, summary: dict) -> dict:
        self._write_report(summary)
        if self.use_wandb:
            wandb.finish()
        return summary

    def _run_random_or_greedy(self):
        saved_paths = 0
        attempts = 0
        total_reactions = 0
        best_qed = 0.0
        while (
            (self.max_paths is None or saved_paths < self.max_paths)
            and attempts < self.max_attempts
            and (self.max_reactions is None or total_reactions < self.max_reactions)
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
                if self.max_reactions is not None and total_reactions >= self.max_reactions:
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
            self._log_search_progress(
                attempts=attempts,
                saved_paths=saved_paths,
                total_reactions=total_reactions,
                last_path_length=reaction_steps,
                initial_qed=initial_qed,
                terminal_qed=path_rows[-1].qed,
                best_qed=best_qed,
            )
        return self._finish(self._summary(attempts, saved_paths, total_reactions, best_qed))

    def _run_exhausted(self):
        saved_paths = 0
        starts_seen = 0
        total_reactions = 0
        best_qed = 0.0

        def save_terminal(path_rows: list[SearchStep], initial_qed: float) -> None:
            nonlocal saved_paths, best_qed
            if self.max_paths is not None and saved_paths >= self.max_paths:
                return
            finalized = []
            for row in path_rows:
                finalized.append(
                    SearchStep(
                        path_id=saved_paths,
                        step=row.step,
                        reactant=row.reactant,
                        template=row.template,
                        product=row.product,
                        qed=row.qed,
                        reward=row.reward,
                        second_reactant=row.second_reactant,
                    )
                )
            self.all_steps.extend(finalized)
            saved_paths += 1
            best_qed = max(best_qed, path_rows[-1].qed)
            self._log_search_progress(
                attempts=starts_seen,
                saved_paths=saved_paths,
                total_reactions=total_reactions,
                last_path_length=len(path_rows) - 1,
                initial_qed=initial_qed,
                terminal_qed=path_rows[-1].qed,
                best_qed=best_qed,
            )

        def dfs(current: str, path_rows: list[SearchStep], initial_qed: float) -> None:
            nonlocal total_reactions, best_qed
            if self._reached_limit(saved_paths, total_reactions):
                return
            step_idx = len(path_rows) - 1
            if step_idx >= self.max_steps:
                save_terminal(path_rows, initial_qed)
                return
            next_actions = self._all_next(current)
            if not next_actions:
                save_terminal(path_rows, initial_qed)
                return
            for template_idx, product, r2, reward in next_actions:
                if self._reached_limit(saved_paths, total_reactions):
                    return
                q = qed(product)
                best_qed = max(best_qed, q)
                total_reactions += 1
                dfs(
                    product,
                    path_rows
                    + [
                        SearchStep(
                            path_id=-1,
                            step=step_idx + 1,
                            reactant=current,
                            template=self._template_name(template_idx),
                            product=product,
                            qed=round(q, 3),
                            reward=reward,
                            second_reactant=r2,
                        )
                    ],
                    initial_qed,
                )

        for start in self.reactant_keys:
            if self._reached_limit(saved_paths, total_reactions):
                break
            if self.max_starts is not None and starts_seen >= self.max_starts:
                break
            if not _valid_smiles(start):
                continue
            starts_seen += 1
            initial_qed = qed(start)
            dfs(
                start,
                [
                    SearchStep(
                        path_id=-1,
                        step=0,
                        reactant=start,
                        template="START",
                        product=start,
                        qed=round(initial_qed, 3),
                        reward=0.0,
                        second_reactant=None,
                    )
                ],
                initial_qed,
            )
        summary = self._summary(starts_seen, saved_paths, total_reactions, best_qed)
        summary["starts_seen"] = starts_seen
        return self._finish(summary)

    def run(self):
        if self.mode == "exhausted_search":
            return self._run_exhausted()
        return self._run_random_or_greedy()


def train(config: dict, experiment_name: str, *, mode: str):
    return SearchRunner(config, experiment_name, mode).run()
