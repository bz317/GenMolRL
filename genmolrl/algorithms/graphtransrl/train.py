"""GraphTransRL trainer for GenMolRL.

GraphTransRL is the graph-transformer RL method for the current Uni-reaction
objective. Start molecules are supplied externally and rewards are per-action
delta-QED.
"""

from __future__ import annotations

import math
import pickle
import random
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED

import wandb
from genmolrl.algorithms.common import init_wandb, run_dir, set_seed
from genmolrl.algorithms.graphtransrl.graph_transformer import GraphTransRLPolicy, batch_from_smiles
from genmolrl.chem.reaction_manager import ReactionManager
from genmolrl.config import resolve_path

STOP_ACTION = "Stop"


def require_graphtransrl_dependencies() -> None:
    if importlib.util.find_spec("torch_geometric") is None:
        raise ImportError(
            "GraphTransRL requires torch_geometric. Install it in the active environment, e.g. "
            "`python -m pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html` "
            "for the current torch==2.3.0+cu121 environment."
        )


def _load_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _reactant_smiles(data: Any) -> list[str]:
    if isinstance(data, dict):
        return [str(k) for k in data.keys()]
    if isinstance(data, (list, tuple, set)):
        return [str(x) for x in data]
    raise ValueError("Reactant file must contain a dict or sequence of SMILES.")


def _qed(smiles: str, *, round_digits: int | None = None) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    value = float(QED.qed(mol))
    return round(value, round_digits) if round_digits is not None else value


@dataclass
class Trajectory:
    start_smiles: str
    final_smiles: str
    actions: list[int | str]
    rewards: list[float]
    qeds: list[float]
    smiles: list[str]

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def episode_len(self) -> int:
        return len([a for a in self.actions if a != STOP_ACTION])

    @property
    def final_delta_qed(self) -> float:
        return float(self.qeds[-1] - self.qeds[0])

    @property
    def max_qed(self) -> float:
        return float(max(self.qeds))


class _BestTrajectoryReplayBuffer:
    """Top-K best trajectories ever sampled, by stored ``total_reward``.

    Stores the SMILES sequence so :class:`GraphTransRL.replay_trajectory` can
    recompute log-probabilities under the current policy without re-running RDKit.
    Duplicates are deduped via ``(start_smiles, tuple(actions))`` so the buffer
    holds *distinct* high-reward modes.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        # entries are (total_reward, start_smiles, actions, smiles_sequence)
        # sorted ascending by total_reward; index 0 is the lowest-reward entry.
        self.entries: list[tuple[float, str, list, list]] = []
        self._keys: set[tuple] = set()

    def __len__(self) -> int:
        return len(self.entries)

    def push(self, total_reward: float, start_smiles: str, actions: list, smiles_sequence: list) -> bool:
        if self.capacity <= 0:
            return False
        key = (start_smiles, tuple(actions))
        if key in self._keys:
            return False
        if len(self.entries) < self.capacity:
            self.entries.append((float(total_reward), str(start_smiles), list(actions), list(smiles_sequence)))
            self._keys.add(key)
            self.entries.sort(key=lambda e: e[0])
            return True
        if total_reward > self.entries[0][0]:
            removed = self.entries.pop(0)
            self._keys.discard((removed[1], tuple(removed[2])))
            self.entries.append((float(total_reward), str(start_smiles), list(actions), list(smiles_sequence)))
            self._keys.add(key)
            self.entries.sort(key=lambda e: e[0])
            return True
        return False

    def sample_one(self) -> tuple[float, str, list, list] | None:
        if not self.entries:
            return None
        return random.choice(self.entries)

    def top_reward(self) -> float:
        return self.entries[-1][0] if self.entries else float("nan")


class StartSampler:
    def __init__(self, train_smiles: list[str], test_smiles: list[str], seed: int):
        if not train_smiles:
            raise ValueError("GraphTransRL training requires at least one training molecule.")
        if not test_smiles:
            raise ValueError("GraphTransRL evaluation requires at least one test molecule.")
        self.train_smiles = list(train_smiles)
        self.test_smiles = list(test_smiles)
        self.rng = random.Random(seed)

    def sample_train(self, batch_size: int) -> list[str]:
        return [self.rng.choice(self.train_smiles) for _ in range(batch_size)]

    def eval_starts(self) -> list[str]:
        return list(self.test_smiles)


class GraphTransRL:
    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config.get("seed", config.get("training", {}).get("seed", 0)))
        set_seed(self.seed)
        dataset = config["dataset"]
        self.train_reactants = _load_pickle(resolve_path(dataset["training_file"]))
        self.test_reactants = _load_pickle(resolve_path(dataset["test_file"]))
        self.templates = _load_pickle(resolve_path(dataset["templates_file"]))
        self.train_smiles = _reactant_smiles(self.train_reactants)
        self.test_smiles = _reactant_smiles(self.test_reactants)
        self.reaction_mode = config.get("reaction_mode", "uni")
        self.masking = config.get("masking", "reaction_valid")
        self.reward_name = config.get("reward", "delta_qed")
        if self.reward_name != "delta_qed":
            raise ValueError("GraphTransRL currently supports reward: delta_qed")
        self.max_episode_len = int(config.get("max_episode_len", config.get("env", {}).get("max_episode_len", 5)))
        self.use_stop_action = bool(config.get("env", {}).get("use_stop_action", True))
        self.qed_round_digits = config.get("env", {}).get("info_qed_round_digits", config.get("env", {}).get("reward_round_digits"))
        manager_source = self.train_reactants if isinstance(self.train_reactants, dict) else {s: None for s in self.train_smiles}
        self.reaction_manager = ReactionManager(self.templates, manager_source)
        if self.reaction_mode == "uni":
            self.reaction_manager.templates = self.reaction_manager.templates_for_mode("uni")
            self.reaction_manager.template_keys = list(self.reaction_manager.templates.keys())
            self.reaction_manager.template_mask_cache.clear()
        elif self.reaction_mode != "bi":
            raise ValueError(f"Unsupported reaction_mode: {self.reaction_mode}")
        self.num_templates = len(self.reaction_manager.templates)
        self.stop_index = self.num_templates
        method_cfg = config.get("graphtransrl", {})
        self.device = torch.device(method_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = GraphTransRLPolicy(
            self.num_templates,
            num_emb=int(method_cfg.get("num_emb", 64)),
            num_layers=int(method_cfg.get("num_layers", 3)),
            num_heads=int(method_cfg.get("num_heads", 2)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(method_cfg.get("learning_rate", config.get("training", {}).get("learning_rate", 1e-4))),
            weight_decay=float(method_cfg.get("weight_decay", 0.0)),
        )
        self.log_z = torch.nn.Parameter(torch.tensor(float(method_cfg.get("init_log_z", 0.0)), device=self.device))
        self.optimizer.add_param_group({"params": [self.log_z], "lr": float(method_cfg.get("log_z_lr", 1e-3))})
        # Optional epsilon-greedy template exploration with linear decay. Defaults
        # preserve previous behavior: constant ``random_action_prob`` (0.0 by default).
        self.random_action_prob = float(method_cfg.get("random_action_prob", 0.0))
        self.random_action_min_prob = float(
            method_cfg.get("random_action_min_prob", self.random_action_prob)
        )
        self.random_action_decay_steps = int(method_cfg.get("random_action_decay_steps", 0))
        self.current_global_step = 0
        # Inverse temperature on the TB target (target = beta * log(R+1)). Default 1.0
        # reproduces the original target. Higher beta sharpens toward high-R trajectories.
        self.reward_beta = float(method_cfg.get("reward_beta", 1.0))
        # Optional best-trajectory replay buffer. Disabled when capacity == 0.
        self.replay_buffer = _BestTrajectoryReplayBuffer(
            int(method_cfg.get("replay_buffer_size", 0))
        )
        self.replay_prob = float(method_cfg.get("replay_prob", 0.0))
        self.sampler = StartSampler(self.train_smiles, self.test_smiles, self.seed)

    def _effective_random_action_prob(self) -> float:
        if self.random_action_prob <= 0.0 or self.random_action_decay_steps <= 0:
            return float(self.random_action_prob)
        progress = self.current_global_step / float(self.random_action_decay_steps)
        progress = max(0.0, min(1.0, progress))
        return float(
            self.random_action_prob
            + (self.random_action_min_prob - self.random_action_prob) * progress
        )

    def _action_mask(self, smiles: str) -> torch.Tensor:
        mask = torch.zeros(self.num_templates + 1, dtype=torch.bool, device=self.device)
        template_mask = self.reaction_manager.get_mask(smiles, kind=self.masking).to(self.device) > 0.5
        mask[: self.num_templates] = template_mask
        if self.use_stop_action:
            mask[self.stop_index] = True
        return mask

    def _masked_logits(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
        graph = batch_from_smiles([smiles], device=self.device)
        logits = self.policy(graph, torch.ones((1, 1), device=self.device))[0]
        mask = self._action_mask(smiles)
        return logits.masked_fill(~mask, -1e9), mask

    def _choose_action(self, logits: torch.Tensor, mask: torch.Tensor, *, greedy: bool) -> tuple[int, torch.Tensor]:
        if greedy:
            action = int(torch.argmax(logits).item())
        else:
            eps = self._effective_random_action_prob()
            if eps > 0.0 and random.random() < eps:
                valid = torch.where(mask)[0]
                action = int(valid[torch.randint(len(valid), (1,), device=self.device)].item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())
        log_prob = F.log_softmax(logits, dim=-1)[action]
        return action, log_prob

    def sample_trajectory(self, start_smiles: str, *, greedy: bool = False, track_grad: bool = True):
        ctx = torch.enable_grad() if track_grad else torch.no_grad()
        with ctx:
            current = str(start_smiles)
            current_qed = _qed(current, round_digits=self.qed_round_digits)
            traj = Trajectory(
                start_smiles=current,
                final_smiles=current,
                actions=[],
                rewards=[],
                qeds=[current_qed],
                smiles=[current],
            )
            log_probs: list[torch.Tensor] = []
            forward_log_flows: list[torch.Tensor] = []

            for _ in range(self.max_episode_len + int(self.use_stop_action)):
                logits, mask = self._masked_logits(current)
                if not bool(mask.any()):
                    break
                action, log_prob = self._choose_action(logits, mask, greedy=greedy)
                log_probs.append(log_prob)
                forward_log_flows.append(torch.logsumexp(logits[mask], dim=0))
                if action == self.stop_index:
                    traj.actions.append(STOP_ACTION)
                    break
                if traj.episode_len >= self.max_episode_len:
                    break
                product = self.reaction_manager.apply_reaction(current, self.reaction_manager.templates[action], None)
                if product is None:
                    traj.actions.append(action)
                    traj.rewards.append(-1.0)
                    break
                next_qed = _qed(product, round_digits=self.qed_round_digits)
                reward = float(next_qed - current_qed)
                traj.actions.append(action)
                traj.rewards.append(reward)
                traj.qeds.append(next_qed)
                traj.smiles.append(product)
                current = product
                current_qed = next_qed
                traj.final_smiles = product
                if traj.episode_len >= self.max_episode_len:
                    break
            if not forward_log_flows:
                forward_log_flows.append(self.log_z * 0.0)
            return traj, log_probs, forward_log_flows

    def replay_trajectory(
        self,
        start_smiles: str,
        actions: list,
        smiles_sequence: list,
        *,
        track_grad: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Recompute log-pf and forward-log-flows for a fixed trajectory under the current policy.

        Used by the best-trajectory replay buffer. The trajectory is faithful to
        the original because the action mask only depends on the current SMILES
        (deterministic), so each ``(action, current_smiles)`` pair is replayable.
        If a stored action becomes infeasible (e.g., template pool changed), the
        replay terminates early and the surviving prefix is used for the loss.
        """
        ctx = torch.enable_grad() if track_grad else torch.no_grad()
        log_probs: list[torch.Tensor] = []
        forward_log_flows: list[torch.Tensor] = []
        with ctx:
            for i, action in enumerate(actions):
                if i >= len(smiles_sequence):
                    break
                current = smiles_sequence[i]
                logits, mask = self._masked_logits(current)
                if not bool(mask.any()):
                    break
                if isinstance(action, str) and action == STOP_ACTION:
                    if not self.use_stop_action or not bool(mask[self.stop_index]):
                        break
                    log_prob = F.log_softmax(logits, dim=-1)[self.stop_index]
                else:
                    a = int(action)
                    if a < 0 or a >= mask.numel() or not bool(mask[a]):
                        break
                    log_prob = F.log_softmax(logits, dim=-1)[a]
                log_probs.append(log_prob)
                forward_log_flows.append(torch.logsumexp(logits[mask], dim=0))
            if not forward_log_flows:
                forward_log_flows.append(self.log_z * 0.0)
        return log_probs, forward_log_flows

    def train_step(self, starts: list[str]) -> dict[str, float]:
        self.policy.train()
        losses = []
        rewards: list[float] = []
        lengths: list[int] = []
        replay_flags: list[float] = []
        for start in starts:
            replay_used = False
            sampled_total_reward: float
            sampled_episode_len: int
            if (
                self.replay_prob > 0.0
                and len(self.replay_buffer) > 0
                and random.random() < self.replay_prob
            ):
                entry = self.replay_buffer.sample_one()
                if entry is not None:
                    stored_reward, replay_start, replay_actions, replay_smiles = entry
                    log_probs, forward_log_flows = self.replay_trajectory(
                        replay_start, replay_actions, replay_smiles, track_grad=True
                    )
                    sampled_total_reward = float(stored_reward)
                    sampled_episode_len = int(
                        len([a for a in replay_actions if a != STOP_ACTION])
                    )
                    replay_used = True
            if not replay_used:
                traj, log_probs, forward_log_flows = self.sample_trajectory(
                    start, greedy=False, track_grad=True
                )
                sampled_total_reward = float(traj.total_reward)
                sampled_episode_len = int(traj.episode_len)
                # Push fresh (on-policy) trajectory into the best-K buffer.
                self.replay_buffer.push(
                    traj.total_reward,
                    traj.start_smiles,
                    list(traj.actions),
                    list(traj.smiles),
                )

            rewards.append(sampled_total_reward)
            lengths.append(sampled_episode_len)
            replay_flags.append(float(replay_used))
            terminal_reward = max(1e-6, sampled_total_reward + 1.0)
            target = torch.tensor(
                self.reward_beta * math.log(terminal_reward), device=self.device
            )
            log_pf = (
                torch.stack(log_probs).sum() if log_probs else self.log_z * 0.0
            )
            flow_term = torch.stack(forward_log_flows).mean()
            losses.append((self.log_z + log_pf - target).pow(2) + 1e-3 * flow_term.pow(2))
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()
        return {
            "train/loss": float(loss.detach().cpu().item()),
            "train/mean_reward": float(np.mean(rewards)),
            "train/mean_ep_length": float(np.mean(lengths)),
            "train/log_z": float(self.log_z.detach().cpu().item()),
            "train/eps_random_action": self._effective_random_action_prob(),
            "train/replay_fraction": float(np.mean(replay_flags)) if replay_flags else 0.0,
            "train/replay_buffer_size": int(len(self.replay_buffer)),
            "train/replay_buffer_top_reward": self.replay_buffer.top_reward(),
            "train/reward_beta": float(self.reward_beta),
        }

    def evaluate(self) -> dict[str, float]:
        self.policy.eval()
        trajectories = [self.sample_trajectory(s, greedy=True, track_grad=False)[0] for s in self.sampler.eval_starts()]
        rewards = [t.total_reward for t in trajectories]
        final_deltas = [t.final_delta_qed for t in trajectories]
        lengths = [t.episode_len for t in trajectories]
        max_qed = max((t.max_qed for t in trajectories), default=float("nan"))
        return {
            "eval/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "eval/avg_delta_qed": float(np.mean(final_deltas)) if final_deltas else 0.0,
            "eval/mean_final_delta_qed": float(np.mean(final_deltas)) if final_deltas else 0.0,
            "eval/max_qed": float(max_qed),
            "eval/mean_ep_length": float(np.mean(lengths)) if lengths else 0.0,
            "eval/n_molecules": len(trajectories),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "log_z": self.log_z.detach().cpu(),
                "config": self.config,
            },
            path,
        )


def train(config: dict, experiment_name: str) -> None:
    require_graphtransrl_dependencies()
    trainer = GraphTransRL(config)
    run = init_wandb(config, "graphtransrl", experiment_name)
    training = config.get("training", {})
    total_steps = int(training.get("total_timesteps", training.get("num_steps", 1000)))
    batch_size = int(training.get("batch_size", 16))
    eval_freq = int(training.get("eval_freq", 1000))
    save_freq = int(training.get("save_freq", eval_freq))
    out_dir = run_dir(run.id if run is not None else experiment_name)

    best_eval = -float("inf")
    for global_step in range(1, total_steps + 1):
        trainer.current_global_step = global_step
        starts = trainer.sampler.sample_train(batch_size)
        metrics = trainer.train_step(starts)
        metrics["train/global_step"] = global_step
        wandb.log(metrics, step=global_step)
        if eval_freq > 0 and (global_step % eval_freq == 0 or global_step == total_steps):
            eval_metrics = trainer.evaluate()
            eval_metrics["train/global_step"] = global_step
            wandb.log(eval_metrics, step=global_step)
            if eval_metrics["eval/mean_reward"] > best_eval:
                best_eval = eval_metrics["eval/mean_reward"]
                trainer.save(out_dir / "best_model.pt")
        if save_freq > 0 and global_step % save_freq == 0:
            trainer.save(out_dir / f"model_step_{global_step}.pt")
    trainer.save(out_dir / "final_model.pt")
    if run is not None:
        run.finish()


__all__ = ["GraphTransRL", "train"]
