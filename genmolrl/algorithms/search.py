"""Non-neural search baselines over the shared GenMolRL chemistry layer."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

import wandb
from genmolrl.chem.datasets import load_pickle
from genmolrl.chem.reaction_manager import BI_TYPE, ReactionManager
from genmolrl.config import project_root, resolve_path
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
            qed_round_digits=config.get("env", {}).get("info_qed_round_digits"),
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
        self.greedy_mode = str(self.search_cfg.get("greedy_mode", "best_action")).lower()
        if self.greedy_mode not in {"best_action", "positive_delta_only"}:
            raise ValueError(
                "search.greedy_mode must be 'best_action' or 'positive_delta_only'. "
                f"Got: {self.greedy_mode}"
            )
        self.overwrite_results = bool(self.search_cfg.get("overwrite_results", True))
        self.seed = int(config.get("training", {}).get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)

        reactant_pool_file = self.dataset.get("test_file")
        if reactant_pool_file is None:
            raise KeyError("dataset.test_file must be set for search baselines")
        reactants = load_pickle(resolve_path(reactant_pool_file))
        templates = load_pickle(resolve_path(self.dataset["templates_file"]))
        all_manager = ReactionManager(templates, reactants)
        templates_for_mode = all_manager.templates_for_mode(config["reaction_mode"])
        self.manager = ReactionManager(templates_for_mode, reactants)
        self.reactant_keys = list(reactants.keys())
        self.result_file = Path(
            self.search_cfg.get(
                "results_file",
                f"runs/{self.mode}_results.txt",
            )
        )
        if not self.result_file.is_absolute():
            self.result_file = project_root() / self.result_file
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
        self.trajectory_summaries: list[dict] = []
        self._traj_count: int = 0
        self._delta_qed_sum: float = 0.0
        self._num_reactions_sum: int = 0
        self._stream_paths: Path | None = None
        self._stream_trajs: Path | None = None
        self._stream_progress_every: int = int(self.search_cfg.get("progress_every_starts", 100))
        # How often to flush+fsync the streaming sidecar files (in paths).
        # Defaults to 1 (every path) so visibility is immediate; set higher for fewer syscalls.
        self._stream_flush_every_paths: int = int(self.search_cfg.get("flush_every_paths", 1))
        # How often to print a path-level stdout heartbeat (in paths).
        self._stream_path_heartbeat_every: int = int(self.search_cfg.get("path_heartbeat_every", 100))
        # How often to print an intra-DFS state-visit heartbeat (in state visits).
        # Critical for bi-mode under masking=reaction_valid where each state visit
        # is expensive enough that no `save_terminal` may fire for hours.
        # Set to 0 to disable.
        self._stream_state_heartbeat_every: int = int(
            self.search_cfg.get("state_heartbeat_every", 100)
        )
        # When set, write a snapshot of the *currently-explored* path-so-far to
        # `<result_file>.inprogress.tmp` every N state visits. This makes the
        # search frontier observable on disk even when no leaf has been saved.
        # Set to 0 to disable.
        self._stream_inprogress_every: int = int(
            self.search_cfg.get("inprogress_snapshot_every", 100)
        )

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
        if self.mode == "greedy_search" and self.greedy_mode == "positive_delta_only" and reward <= 0:
            return None
        return idx, product, r2, reward

    def _all_next(self, current: str) -> list[tuple[int, str, str | None, float]]:
        candidates = []
        for idx in self._valid_template_indices(current):
            for product, r2, reward in self._candidate_products(current, idx):
                candidates.append((idx, product, r2, reward))
        return candidates

    # Field order shared across in-memory and streaming writers
    _STEP_FIELDS = (
        "path_id",
        "step",
        "reactant",
        "template",
        "product",
        "qed",
        "reward",
        "second_reactant",
    )
    _TRAJ_FIELDS = (
        "path_id",
        "start_smiles",
        "final_smiles",
        "initial_qed",
        "final_qed",
        "max_qed",
        "delta_qed",
        "num_reactions",
    )

    @staticmethod
    def _format_step_row(step: SearchStep) -> str:
        row = step.__dict__
        return "\t".join("" if row[k] is None else str(row[k]) for k in SearchRunner._STEP_FIELDS) + "\n"

    @staticmethod
    def _format_traj_row(item: dict) -> str:
        return "\t".join(str(item.get(k, "")) for k in SearchRunner._TRAJ_FIELDS) + "\n"

    def _write_report(self, summary: dict) -> None:
        with self.result_file.open("w", encoding="utf-8") as f:
            f.write("# GenMolRL search results\n\n")
            f.write("[summary]\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            if self.trajectory_summaries:
                f.write("\n[trajectories]\n")
                f.write("\t".join(self._TRAJ_FIELDS) + "\n")
                for item in self.trajectory_summaries:
                    f.write(self._format_traj_row(item))
            f.write("\n[steps]\n")
            f.write("\t".join(self._STEP_FIELDS) + "\n")
            for step in self.all_steps:
                f.write(self._format_step_row(step))

    def _consolidate_streamed_report(self, summary: dict) -> None:
        """Stitch streamed `.trajectories.tmp` and `.steps.tmp` sidecar files into the final report."""
        traj_tmp = self._stream_trajs
        steps_tmp = self._stream_paths
        with self.result_file.open("w", encoding="utf-8") as f:
            f.write("# GenMolRL search results\n\n")
            f.write("[summary]\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n[trajectories]\n")
            f.write("\t".join(self._TRAJ_FIELDS) + "\n")
            if traj_tmp is not None and traj_tmp.exists():
                with traj_tmp.open("r", encoding="utf-8") as tin:
                    for line in tin:
                        f.write(line)
            f.write("\n[steps]\n")
            f.write("\t".join(self._STEP_FIELDS) + "\n")
            if steps_tmp is not None and steps_tmp.exists():
                with steps_tmp.open("r", encoding="utf-8") as sin:
                    for line in sin:
                        f.write(line)
        if traj_tmp is not None:
            traj_tmp.unlink(missing_ok=True)
        if steps_tmp is not None:
            steps_tmp.unlink(missing_ok=True)
        # The frontier snapshot is only useful while the search is running.
        inprogress = self.result_file.with_suffix(".inprogress.tmp")
        inprogress.unlink(missing_ok=True)

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
        try:
            result_file = str(self.result_file.relative_to(project_root()))
        except ValueError:
            result_file = str(self.result_file)
        n = self._traj_count
        mean_delta = self._delta_qed_sum / n if n else 0.0
        mean_reactions = self._num_reactions_sum / n if n else 0.0
        return {
            "attempts": attempts,
            "saved_paths": saved_paths,
            "total_reactions": total_reactions,
            "max_qed": best_qed,
            "best_qed": best_qed,
            "avg_delta_qed": mean_delta,
            "mean_delta_qed": mean_delta,
            "sum_delta_qed": self._delta_qed_sum,
            "num_start_molecules": n,
            "avg_num_reactions": mean_reactions,
            "max_episode_len": self.max_steps,
            "greedy_mode": self.greedy_mode if self.mode == "greedy_search" else "",
            "results_file": result_file,
        }

    def _record_trajectory(self, path_rows: list[SearchStep], initial_qed: float, path_id: int) -> dict:
        final_row = path_rows[-1]
        final_qed = float(qed(final_row.product))
        path_max_qed = max(float(row.qed) for row in path_rows)
        summary = {
            "path_id": path_id,
            "start_smiles": path_rows[0].product,
            "final_smiles": final_row.product,
            "initial_qed": round(float(initial_qed), 6),
            "final_qed": round(final_qed, 6),
            "max_qed": round(path_max_qed, 6),
            "delta_qed": round(final_qed - float(initial_qed), 6),
            "num_reactions": len(path_rows) - 1,
        }
        self._traj_count += 1
        self._delta_qed_sum += float(summary["delta_qed"])
        self._num_reactions_sum += int(summary["num_reactions"])
        return summary

    def _finish(self, summary: dict) -> dict:
        self._write_report(summary)
        if self.use_wandb:
            wandb.finish()
        return summary

    def _run_random_or_greedy(self):
        saved_paths = 0
        starts_seen = 0
        total_reactions = 0
        best_qed = 0.0
        for current in self.reactant_keys:
            if self.max_starts is not None and starts_seen >= self.max_starts:
                break
            if self._reached_limit(saved_paths, total_reactions):
                break
            if not _valid_smiles(current):
                continue
            starts_seen += 1
            initial_qed = qed(current)
            best_qed = max(best_qed, initial_qed)
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
            self.all_steps.extend(path_rows)
            self.trajectory_summaries.append(
                self._record_trajectory(path_rows, initial_qed, saved_paths)
            )
            saved_paths += 1
            self._log_search_progress(
                attempts=starts_seen,
                saved_paths=saved_paths,
                total_reactions=total_reactions,
                last_path_length=reaction_steps,
                initial_qed=initial_qed,
                terminal_qed=path_rows[-1].qed,
                best_qed=best_qed,
            )
        summary = self._summary(starts_seen, saved_paths, total_reactions, best_qed)
        summary["starts_seen"] = starts_seen
        return self._finish(summary)

    def _run_exhausted(self):
        saved_paths = 0
        starts_seen = 0
        total_reactions = 0
        best_qed = 0.0
        state_visits = 0
        last_heartbeat_t = 0.0

        # Sidecar files for streaming output. Created fresh per run so a partial run still
        # leaves a readable on-disk record (concatenated into the main report on completion).
        # Writes happen inside `save_terminal` as soon as each path is finalised — disk
        # therefore grows continuously regardless of how deep/wide a single start's DFS is.
        self._stream_paths = self.result_file.with_suffix(".steps.tmp")
        self._stream_trajs = self.result_file.with_suffix(".trajectories.tmp")
        # Frontier snapshot: holds the *current* trajectory being explored, even
        # before any leaf has been saved. Overwritten on every snapshot.
        inprogress_file = self.result_file.with_suffix(".inprogress.tmp")
        self._stream_paths.parent.mkdir(parents=True, exist_ok=True)
        for p in (self._stream_paths, self._stream_trajs, inprogress_file):
            if p.exists():
                p.unlink()

        run_start = time.monotonic()
        # File handles published into the save_terminal closure via this dict.
        # Using a mutable container lets us define save_terminal/dfs before the `with`
        # block opens the files, keeping the existing dfs recursion intact.
        handles: dict[str, "object"] = {"steps": None, "traj": None}

        def write_inprogress(path_rows: list[SearchStep]) -> None:
            try:
                with inprogress_file.open("w", encoding="utf-8") as f:
                    f.write(
                        f"# in-progress frontier (live, last update t={time.monotonic() - run_start:.0f}s)\n"
                    )
                    f.write(
                        f"# state_visits={state_visits} saved_paths={saved_paths} "
                        f"total_reactions={total_reactions} starts_seen={starts_seen}\n"
                    )
                    f.write("\t".join(self._STEP_FIELDS) + "\n")
                    for row in path_rows:
                        f.write(self._format_step_row(row))
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                # Don't let snapshot IO failures kill the search.
                pass

        def save_terminal(path_rows: list[SearchStep], initial_qed: float) -> None:
            nonlocal saved_paths, best_qed
            if self.max_paths is not None and saved_paths >= self.max_paths:
                return
            finalized = [
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
                for row in path_rows
            ]
            summary = self._record_trajectory(finalized, initial_qed, saved_paths)
            f_steps = handles["steps"]
            f_traj = handles["traj"]
            f_steps.writelines(self._format_step_row(s) for s in finalized)
            f_traj.write(self._format_traj_row(summary))
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
            if self._stream_flush_every_paths > 0 and saved_paths % self._stream_flush_every_paths == 0:
                f_steps.flush()
                f_traj.flush()
            if (
                self._stream_path_heartbeat_every > 0
                and saved_paths % self._stream_path_heartbeat_every == 0
            ):
                elapsed = time.monotonic() - run_start
                rate = saved_paths / elapsed if elapsed > 0 else 0.0
                print(
                    f"[exhausted/path] starts={starts_seen} saved_paths={saved_paths} "
                    f"total_reactions={total_reactions} best_qed={best_qed:.3f} "
                    f"elapsed={elapsed:.0f}s rate={rate:.1f} paths/s",
                    flush=True,
                )

        def dfs(current: str, path_rows: list[SearchStep], initial_qed: float) -> None:
            nonlocal total_reactions, best_qed, state_visits, last_heartbeat_t
            if self._reached_limit(saved_paths, total_reactions):
                return
            step_idx = len(path_rows) - 1
            state_visits += 1
            # State-visit heartbeat. Fires *before* the expensive _all_next call so the
            # log shows the frontier even when no `save_terminal` has fired in hours.
            if (
                self._stream_state_heartbeat_every > 0
                and state_visits % self._stream_state_heartbeat_every == 0
            ):
                elapsed = time.monotonic() - run_start
                rate = state_visits / elapsed if elapsed > 0 else 0.0
                print(
                    f"[exhausted/state] state_visits={state_visits} depth={step_idx} "
                    f"saved_paths={saved_paths} total_reactions={total_reactions} "
                    f"starts_seen={starts_seen} elapsed={elapsed:.0f}s "
                    f"rate={rate:.1f} states/s current={current[:60]}",
                    flush=True,
                )
                last_heartbeat_t = elapsed
            if (
                self._stream_inprogress_every > 0
                and state_visits % self._stream_inprogress_every == 0
            ):
                write_inprogress(path_rows)
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

        try:
            with self._stream_paths.open("w", encoding="utf-8") as f_steps, \
                    self._stream_trajs.open("w", encoding="utf-8") as f_traj:
                handles["steps"] = f_steps
                handles["traj"] = f_traj
                for start in self.reactant_keys:
                    if self._reached_limit(saved_paths, total_reactions):
                        break
                    if self.max_starts is not None and starts_seen >= self.max_starts:
                        break
                    if not _valid_smiles(start):
                        continue
                    starts_seen += 1
                    initial_qed = qed(start)
                    paths_before = saved_paths
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
                    paths_for_start = saved_paths - paths_before
                    # Per-start fsync gives a durable checkpoint on every completed start.
                    f_steps.flush()
                    f_traj.flush()
                    os.fsync(f_steps.fileno())
                    os.fsync(f_traj.fileno())
                    if self._stream_progress_every > 0 and starts_seen % self._stream_progress_every == 0:
                        elapsed = time.monotonic() - run_start
                        rate = saved_paths / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[exhausted/start] starts={starts_seen} last_paths={paths_for_start} "
                            f"saved_paths={saved_paths} total_reactions={total_reactions} "
                            f"best_qed={best_qed:.3f} elapsed={elapsed:.0f}s rate={rate:.1f} paths/s",
                            flush=True,
                        )
        except BaseException:
            # Surface what we have on disk before re-raising so SIGTERM/OOM still leave a usable trail.
            print(
                f"[exhausted] aborted after starts={starts_seen} saved_paths={saved_paths} "
                f"total_reactions={total_reactions}; partial streams left at "
                f"{self._stream_paths} and {self._stream_trajs}",
                flush=True,
            )
            raise

        summary = self._summary(starts_seen, saved_paths, total_reactions, best_qed)
        summary["starts_seen"] = starts_seen
        self._consolidate_streamed_report(summary)
        if self.use_wandb:
            wandb.finish()
        return summary

    def run(self):
        if self.mode == "exhausted_search":
            return self._run_exhausted()
        return self._run_random_or_greedy()


def train(config: dict, experiment_name: str, *, mode: str):
    return SearchRunner(config, experiment_name, mode).run()
