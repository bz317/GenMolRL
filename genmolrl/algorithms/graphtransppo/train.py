"""GraphTransPPO trainer: PPO on the graph-transformer policy.

The trainer collects on-policy rollouts directly through
:class:`ReactionManager` (the same path GraphTransRL uses), so it does not
depend on the gym ``MoleculeDesignEnv`` wiring. The clipped PPO surrogate is
applied with a per-state value baseline produced by the new value head on
:class:`GraphTransPPOPolicy`.

Key design points:

* Rollouts are flat lists of transitions ``(smiles, action, log_pi_old,
  value, reward, done, mask)``. Trajectory boundaries are encoded in
  ``done``; the GAE recursion masks future advantages on ``done`` and uses a
  single bootstrap ``last_value`` for the trailing partial episode.
* Episode termination causes (Stop, max_episode_len, invalid reaction, no
  feasible mask) all set ``done=True``. We never record a "forced-Stop"
  transition when the trajectory is already at ``max_episode_len`` because
  the action distribution is degenerate there; instead the previous
  reaction transition is marked ``done`` and a fresh start molecule is
  drawn.
* The action mask depends only on the SMILES, so masks captured at rollout
  time can be reused exactly during the multi-epoch update.
"""

from __future__ import annotations

import importlib.util
import math
import pickle
import random
from collections import deque
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
from genmolrl.algorithms.graphtransppo.policy import GraphTransPPOPolicy
from genmolrl.algorithms.graphtransrl.graph_transformer import batch_from_smiles
from genmolrl.chem.reaction_manager import ReactionManager
from genmolrl.config import resolve_path

STOP_ACTION = "Stop"


def require_graphtransppo_dependencies() -> None:
    if importlib.util.find_spec("torch_geometric") is None:
        raise ImportError(
            "GraphTransPPO requires torch_geometric. Install it in the active "
            "environment, e.g. `python -m pip install torch-geometric "
            "-f https://data.pyg.org/whl/torch-2.3.0+cu121.html` for the "
            "current torch==2.3.0+cu121 environment."
        )


def _load_pickle(path: str | Path) -> Any:
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
class _Transition:
    smiles: str
    action: int
    log_pi_old: float
    value: float
    reward: float
    done: bool
    mask: torch.Tensor  # CPU bool tensor of shape [num_templates + 1]


class _StartSampler:
    """Samples training start molecules and exposes the test set for eval."""

    def __init__(self, train_smiles: list[str], test_smiles: list[str], seed: int):
        if not train_smiles:
            raise ValueError("GraphTransPPO training requires at least one training molecule.")
        if not test_smiles:
            raise ValueError("GraphTransPPO evaluation requires at least one test molecule.")
        self.train_smiles = list(train_smiles)
        self.test_smiles = list(test_smiles)
        self.rng = random.Random(seed)

    def sample_train(self) -> str:
        return self.rng.choice(self.train_smiles)

    def eval_starts(self) -> list[str]:
        return list(self.test_smiles)


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var_y = float(np.var(returns))
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var_y)


class GraphTransPPO:
    def __init__(self, config: dict):
        self.config = config
        training_cfg = config.get("training", {})
        self.seed = int(config.get("seed", training_cfg.get("seed", 0)))
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
            raise ValueError("GraphTransPPO currently supports reward: delta_qed")
        env_cfg = config.get("env", {})
        self.max_episode_len = int(
            config.get("max_episode_len", env_cfg.get("max_episode_len", 5))
        )
        self.use_stop_action = bool(env_cfg.get("use_stop_action", True))
        self.qed_round_digits = env_cfg.get(
            "info_qed_round_digits", env_cfg.get("reward_round_digits")
        )
        self.invalid_reaction_penalty = float(
            env_cfg.get("invalid_reaction_penalty", -1.0)
        )

        manager_source = (
            self.train_reactants
            if isinstance(self.train_reactants, dict)
            else {s: None for s in self.train_smiles}
        )
        self.reaction_manager = ReactionManager(self.templates, manager_source)
        if self.reaction_mode == "uni":
            self.reaction_manager.templates = self.reaction_manager.templates_for_mode("uni")
            self.reaction_manager.template_keys = list(self.reaction_manager.templates.keys())
            self.reaction_manager.template_mask_cache.clear()
        elif self.reaction_mode != "bi":
            raise ValueError(f"Unsupported reaction_mode: {self.reaction_mode}")

        self.num_templates = len(self.reaction_manager.templates)
        self.stop_index = self.num_templates
        self.action_dim = self.num_templates + 1

        method_cfg = config.get("graphtransppo", {})
        self.device = torch.device(
            method_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy = GraphTransPPOPolicy(
            self.num_templates,
            num_emb=int(method_cfg.get("num_emb", 64)),
            num_layers=int(method_cfg.get("num_layers", 3)),
            num_heads=int(method_cfg.get("num_heads", 2)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(method_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(method_cfg.get("weight_decay", 0.0)),
            eps=float(method_cfg.get("adam_eps", 1e-5)),
        )

        # PPO knobs
        self.gamma = float(method_cfg.get("gamma", 0.99))
        self.gae_lambda = float(method_cfg.get("gae_lambda", 0.95))
        self.clip_range = float(method_cfg.get("clip_range", 0.2))
        clip_vf = method_cfg.get("clip_range_vf", None)
        self.clip_range_vf = float(clip_vf) if clip_vf is not None else None
        self.vf_coef = float(method_cfg.get("vf_coef", 0.5))
        self.ent_coef = float(method_cfg.get("ent_coef", 0.0))
        self.max_grad_norm = float(method_cfg.get("max_grad_norm", 0.5))
        self.target_kl = method_cfg.get("target_kl", None)
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)
        self.normalize_advantage = bool(method_cfg.get("normalize_advantage", True))

        # Rollout / update sizes
        self.n_steps = int(method_cfg.get("n_steps", training_cfg.get("n_steps", 2048)))
        self.minibatch = int(method_cfg.get("batch_size", training_cfg.get("batch_size", 64)))
        self.n_epochs = int(method_cfg.get("n_epochs", 10))

        self.sampler = _StartSampler(self.train_smiles, self.test_smiles, self.seed)

        # Rollout state carried across collect_rollout calls so that very long
        # rollouts can span natural episode boundaries without truncation bias.
        self._current_smiles: str | None = None
        self._current_react_steps: int = 0

        # Rolling stats over the last 100 episodes (PPO/A2C-compatible logging).
        self._ep_reward_window: deque[float] = deque(maxlen=100)
        self._ep_length_window: deque[int] = deque(maxlen=100)
        self._total_episodes: int = 0

    # ------------------------------------------------------------------
    # Action masking + policy evaluation primitives
    # ------------------------------------------------------------------

    def _action_mask(self, smiles: str, *, force_stop: bool = False) -> torch.Tensor:
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        if not force_stop:
            template_mask = (
                self.reaction_manager.get_mask(smiles, kind=self.masking).to(self.device) > 0.5
            )
            mask[: self.num_templates] = template_mask
        if self.use_stop_action:
            mask[self.stop_index] = True
        return mask

    def _forward_single(
        self, smiles: str, *, force_stop: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(masked_logits, value_scalar, mask)`` for one SMILES."""
        graph = batch_from_smiles([smiles], device=self.device)
        cond = torch.ones((1, 1), device=self.device)
        logits, value = self.policy(graph, cond)
        mask = self._action_mask(smiles, force_stop=force_stop)
        masked = logits[0].masked_fill(~mask, -1e9)
        return masked, value[0], mask

    def _forward_batch(
        self, smiles_list: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, values)`` for a batch of SMILES (no masking)."""
        graph = batch_from_smiles(smiles_list, device=self.device)
        cond = torch.ones((len(smiles_list), 1), device=self.device)
        return self.policy(graph, cond)

    # ------------------------------------------------------------------
    # On-policy rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, n_steps: int) -> tuple[list[_Transition], float]:
        """Collect ``n_steps`` transitions and return ``(transitions, last_value)``.

        ``last_value`` is the bootstrap V(s_T) for the trailing transition if
        the rollout ended mid-episode; 0.0 if it ended exactly on a ``done``.
        """
        self.policy.eval()
        transitions: list[_Transition] = []
        if self._current_smiles is None:
            self._current_smiles = self.sampler.sample_train()
            self._current_react_steps = 0

        ep_reward = 0.0
        ep_length = 0
        steps_taken = 0

        with torch.no_grad():
            while steps_taken < n_steps:
                current = self._current_smiles
                react_steps = self._current_react_steps

                # Episodes ending purely because we reached the budget but the
                # env still has feasible reactions: with use_stop_action this
                # is reachable; without it we just terminate.
                at_max = react_steps >= self.max_episode_len
                if at_max and not self.use_stop_action:
                    if transitions:
                        transitions[-1].done = True
                    self._reset_episode(ep_reward, ep_length)
                    ep_reward, ep_length = 0.0, 0
                    continue

                masked_logits, value, mask = self._forward_single(
                    current, force_stop=at_max
                )
                if not bool(mask.any()):
                    # Nothing legal: end the current episode.
                    if transitions:
                        transitions[-1].done = True
                    self._reset_episode(ep_reward, ep_length)
                    ep_reward, ep_length = 0.0, 0
                    continue

                dist = torch.distributions.Categorical(logits=masked_logits)
                action_t = dist.sample()
                action = int(action_t.item())
                log_pi_old = float(dist.log_prob(action_t).item())
                value_f = float(value.item())

                done = False
                reward = 0.0
                if action == self.stop_index:
                    done = True
                else:
                    product = self.reaction_manager.apply_reaction(
                        current, self.reaction_manager.templates[action], None
                    )
                    if product is None:
                        # The mask is structural; rdkit can still fail to apply
                        # (kekulization etc.). Treat as a hard penalty + done.
                        reward = self.invalid_reaction_penalty
                        done = True
                    else:
                        next_qed = _qed(product, round_digits=self.qed_round_digits)
                        prev_qed = _qed(current, round_digits=self.qed_round_digits)
                        reward = float(next_qed - prev_qed)
                        self._current_smiles = product
                        self._current_react_steps = react_steps + 1
                        if self._current_react_steps >= self.max_episode_len:
                            # Truncate: no further decisions will be recorded
                            # this episode. Bootstrap zero on done is a small
                            # bias but keeps the implementation flat.
                            done = True

                transitions.append(
                    _Transition(
                        smiles=current,
                        action=action,
                        log_pi_old=log_pi_old,
                        value=value_f,
                        reward=reward,
                        done=done,
                        mask=mask.detach().to("cpu"),
                    )
                )
                steps_taken += 1
                ep_reward += reward
                ep_length += 1

                if done:
                    self._reset_episode(ep_reward, ep_length)
                    ep_reward, ep_length = 0.0, 0

        # Bootstrap value for the trailing partial episode (last transition
        # not done): use V(current_smiles).
        last_value = 0.0
        if transitions and not transitions[-1].done:
            with torch.no_grad():
                _, v, _ = self._forward_single(self._current_smiles, force_stop=False)
                last_value = float(v.item())
        return transitions, last_value

    def _reset_episode(self, ep_reward: float, ep_length: int) -> None:
        if ep_length > 0:
            self._ep_reward_window.append(float(ep_reward))
            self._ep_length_window.append(int(ep_length))
            self._total_episodes += 1
        self._current_smiles = self.sampler.sample_train()
        self._current_react_steps = 0

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rollout: list[_Transition],
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(rollout)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            non_terminal = 0.0 if rollout[t].done else 1.0
            next_value = (
                last_value if t == n - 1 else (0.0 if rollout[t].done else rollout[t + 1].value)
            )
            delta = rollout[t].reward + self.gamma * next_value * non_terminal - rollout[t].value
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        values = np.array([tr.value for tr in rollout], dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def ppo_update(
        self,
        rollout: list[_Transition],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        self.policy.train()
        n = len(rollout)
        old_values = np.array([tr.value for tr in rollout], dtype=np.float32)
        adv_norm = advantages.copy()
        if self.normalize_advantage and n > 1:
            adv_norm = (adv_norm - adv_norm.mean()) / (adv_norm.std() + 1e-8)

        smiles_all = [tr.smiles for tr in rollout]
        actions_all = np.array([tr.action for tr in rollout], dtype=np.int64)
        log_pi_old_all = np.array([tr.log_pi_old for tr in rollout], dtype=np.float32)
        masks_all = torch.stack([tr.mask for tr in rollout])  # CPU

        idx = np.arange(n)
        last_kl = 0.0
        loss_acc = []
        pg_acc = []
        v_acc = []
        ent_acc = []
        clip_frac_acc = []
        kl_acc = []
        epochs_done = 0
        early_stopped = False

        for epoch in range(self.n_epochs):
            np.random.shuffle(idx)
            kl_epoch = []
            for start in range(0, n, self.minibatch):
                mb = idx[start : start + self.minibatch]
                if len(mb) == 0:
                    continue
                mb_smiles = [smiles_all[i] for i in mb]
                mb_actions = torch.as_tensor(actions_all[mb], device=self.device)
                mb_log_pi_old = torch.as_tensor(log_pi_old_all[mb], device=self.device)
                mb_adv = torch.as_tensor(adv_norm[mb], device=self.device)
                mb_ret = torch.as_tensor(returns[mb], device=self.device)
                mb_old_v = torch.as_tensor(old_values[mb], device=self.device)
                mb_mask = masks_all[mb].to(self.device)

                logits, values = self._forward_batch(mb_smiles)
                logits = logits.masked_fill(~mb_mask, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                log_pi = dist.log_prob(mb_actions)

                ratio = torch.exp(log_pi - mb_log_pi_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                if self.clip_range_vf is None:
                    v_loss = F.mse_loss(values, mb_ret)
                else:
                    v_clipped = mb_old_v + torch.clamp(
                        values - mb_old_v, -self.clip_range_vf, self.clip_range_vf
                    )
                    v_loss = torch.max((values - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                entropy = dist.entropy().mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_pi - mb_log_pi_old
                    # Schulman approximation: mean[(exp(r) - 1) - r] is an
                    # unbiased low-variance KL estimator for small KL.
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).item()
                    clip_frac = float((torch.abs(ratio - 1.0) > self.clip_range).float().mean().item())

                kl_epoch.append(approx_kl)
                loss_acc.append(float(loss.detach().cpu().item()))
                pg_acc.append(float(pg_loss.detach().cpu().item()))
                v_acc.append(float(v_loss.detach().cpu().item()))
                ent_acc.append(float(entropy.detach().cpu().item()))
                clip_frac_acc.append(clip_frac)
                kl_acc.append(approx_kl)

            epochs_done = epoch + 1
            mean_kl_epoch = float(np.mean(kl_epoch)) if kl_epoch else 0.0
            last_kl = mean_kl_epoch
            if self.target_kl is not None and mean_kl_epoch > 1.5 * self.target_kl:
                early_stopped = True
                break

        ev = _explained_variance(old_values, returns.astype(np.float32))
        return {
            "train/loss": float(np.mean(loss_acc)) if loss_acc else 0.0,
            "train/policy_loss": float(np.mean(pg_acc)) if pg_acc else 0.0,
            "train/value_loss": float(np.mean(v_acc)) if v_acc else 0.0,
            "train/entropy": float(np.mean(ent_acc)) if ent_acc else 0.0,
            "train/approx_kl": float(np.mean(kl_acc)) if kl_acc else 0.0,
            "train/clip_fraction": float(np.mean(clip_frac_acc)) if clip_frac_acc else 0.0,
            "train/epochs_done": float(epochs_done),
            "train/early_stop_kl": float(last_kl),
            "train/early_stopped": float(1.0 if early_stopped else 0.0),
            "train/explained_variance": float(ev),
            "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ------------------------------------------------------------------
    # Greedy evaluation over the test set
    # ------------------------------------------------------------------

    def _greedy_trajectory(self, start_smiles: str) -> tuple[float, float, int, float]:
        self.policy.eval()
        current = str(start_smiles)
        start_qed = _qed(current, round_digits=self.qed_round_digits)
        max_qed = start_qed
        total_reward = 0.0
        react_steps = 0
        with torch.no_grad():
            for _ in range(self.max_episode_len + int(self.use_stop_action)):
                at_max = react_steps >= self.max_episode_len
                if at_max and not self.use_stop_action:
                    break
                masked_logits, _, mask = self._forward_single(current, force_stop=at_max)
                if not bool(mask.any()):
                    break
                action = int(torch.argmax(masked_logits).item())
                if action == self.stop_index:
                    break
                product = self.reaction_manager.apply_reaction(
                    current, self.reaction_manager.templates[action], None
                )
                if product is None:
                    total_reward += self.invalid_reaction_penalty
                    break
                next_qed = _qed(product, round_digits=self.qed_round_digits)
                total_reward += float(next_qed - _qed(current, round_digits=self.qed_round_digits))
                current = product
                react_steps += 1
                max_qed = max(max_qed, next_qed)
        final_qed = _qed(current, round_digits=self.qed_round_digits)
        return total_reward, final_qed - start_qed, react_steps, max_qed

    def evaluate(self) -> dict[str, float]:
        self.policy.eval()
        rewards: list[float] = []
        deltas: list[float] = []
        lengths: list[int] = []
        max_qeds: list[float] = []
        for s in self.sampler.eval_starts():
            r, d, l, mq = self._greedy_trajectory(s)
            rewards.append(r)
            deltas.append(d)
            lengths.append(l)
            max_qeds.append(mq)
        if not rewards:
            return {
                "eval/mean_reward": 0.0,
                "eval/avg_delta_qed": 0.0,
                "eval/mean_final_delta_qed": 0.0,
                "eval/mean_ep_length": 0.0,
                "eval/max_qed": float("nan"),
                "eval/n_molecules": 0,
            }
        return {
            "eval/mean_reward": float(np.mean(rewards)),
            "eval/avg_delta_qed": float(np.mean(deltas)),
            "eval/mean_final_delta_qed": float(np.mean(deltas)),
            "eval/mean_ep_length": float(np.mean(lengths)),
            "eval/max_qed": float(np.max(max_qeds)),
            "eval/n_molecules": len(rewards),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "config": self.config,
            },
            path,
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def train(config: dict, experiment_name: str) -> None:
    require_graphtransppo_dependencies()
    trainer = GraphTransPPO(config)
    run = init_wandb(config, "graphtransppo", experiment_name)

    training_cfg = config.get("training", {})
    method_cfg = config.get("graphtransppo", {})
    total_timesteps = int(training_cfg.get("total_timesteps", 1_000_000))
    eval_freq = int(training_cfg.get("eval_freq", 10_000))
    save_freq = int(training_cfg.get("save_freq", 100_000))
    n_steps = int(method_cfg.get("n_steps", training_cfg.get("n_steps", 2048)))

    out_dir = run_dir(run.id if run is not None else experiment_name)
    best_eval = -float("inf")

    global_step = 0
    last_eval_bucket = -1
    last_save_bucket = -1
    while global_step < total_timesteps:
        rollout, last_value = trainer.collect_rollout(n_steps)
        advantages, returns = trainer.compute_gae(rollout, last_value)
        update_metrics = trainer.ppo_update(rollout, advantages, returns)
        global_step += len(rollout)

        rewards = [tr.reward for tr in rollout]
        rollout_metrics = {
            "train/global_step": global_step,
            "train/rollout_steps": float(len(rollout)),
            "train/rollout_mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "train/mean_reward": float(np.mean(trainer._ep_reward_window))
            if trainer._ep_reward_window
            else 0.0,
            "train/mean_ep_length": float(np.mean(trainer._ep_length_window))
            if trainer._ep_length_window
            else 0.0,
            "train/total_episodes": float(trainer._total_episodes),
        }
        wandb.log({**rollout_metrics, **update_metrics}, step=global_step)

        bucket = global_step // eval_freq
        if eval_freq > 0 and bucket > last_eval_bucket:
            last_eval_bucket = bucket
            eval_metrics = trainer.evaluate()
            eval_metrics["train/global_step"] = global_step
            wandb.log(eval_metrics, step=global_step)
            if eval_metrics["eval/mean_reward"] > best_eval:
                best_eval = eval_metrics["eval/mean_reward"]
                trainer.save(out_dir / "best_model.pt")

        sbucket = global_step // save_freq
        if save_freq > 0 and sbucket > last_save_bucket:
            last_save_bucket = sbucket
            trainer.save(out_dir / f"model_step_{global_step}.pt")

    final_eval = trainer.evaluate()
    final_eval["train/global_step"] = global_step
    wandb.log(final_eval, step=global_step)
    trainer.save(out_dir / "final_model.pt")
    if run is not None:
        run.finish()


__all__ = ["GraphTransPPO", "train"]
