"""Self-contained PGFS-style TD3 agent."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim

from genmolrl.algorithms.td3.constants import TD3_UNI_DISCRETE_ACTION_DESIGN
from genmolrl.algorithms.td3.mask_kind import td3_template_mask_kind
from genmolrl.algorithms.td3.models import ActorNetwork, ActorNetworkUniDiscrete, CriticNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loss_value_for_checkpoint(loss):
    if loss is None:
        return None
    if isinstance(loss, torch.Tensor):
        return float(loss.detach().cpu().item())
    return float(loss)


class TD3Agent:
    def __init__(
        self,
        env,
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_std=0.1,
        noise_clip=0.2,
        policy_freq=2,
        temperature_start=1.0,
        temperature_end=0.1,
        start_timesteps=3000,
        max_timesteps=1000000,
        template_mask_kind: str | None = None,
        entropy_regularization: bool = False,
        entropy_alpha: float = 0.2,
    ):
        self.env = env
        self.state_dim = env.unwrapped.observation_space.shape[0]
        self.template_dim = env.unwrapped.action_space.n
        obs_dim = env.unwrapped.observation_space.shape[0]
        self.discrete_uni = getattr(env.unwrapped, "action_design", "") == TD3_UNI_DISCRETE_ACTION_DESIGN
        self.continuous_r2_dim = 0 if self.discrete_uni else obs_dim

        if self.discrete_uni:
            self.actor = ActorNetworkUniDiscrete(self.state_dim, self.template_dim).to(device)
        else:
            self.actor = ActorNetwork(self.state_dim, self.template_dim, obs_dim).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1 = CriticNetwork(self.state_dim, self.template_dim, self.continuous_r2_dim).to(device)
        self.critic2 = CriticNetwork(self.state_dim, self.template_dim, self.continuous_r2_dim).to(device)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.temperature = temperature_start
        self.temperature_end = temperature_end
        decay_steps = max(1, max_timesteps - start_timesteps)
        self.temperature_decay = math.pow((temperature_end / temperature_start), (1 / decay_steps))
        self.max_action = self.env.unwrapped.observation_space.high[0]
        self.total_it = 0
        self.actor_loss = None
        self.critic_loss = None
        self._template_mask_kind_override = template_mask_kind

        # Opt-in SAC-discrete-style soft actor-critic update. When enabled, the
        # actor is treated as a categorical distribution (masked softmax),
        # trained to maximize ``E_{a~π}[Q(s,a) - α log π(a|s)]``, and the
        # critic uses a soft Bellman backup with the same entropy bonus.
        # Restricted to ``reaction_mode == 'uni'`` since the all-actions Q
        # evaluation assumes R2 = 0 for every action.
        self.entropy_regularization = bool(entropy_regularization)
        self.entropy_alpha = float(entropy_alpha)
        if self.entropy_regularization:
            reaction_mode = getattr(env.unwrapped, "reaction_mode", "uni")
            if reaction_mode != "uni":
                raise ValueError(
                    "td3.entropy_regularization=True is only supported for reaction_mode=uni "
                    "(the entropy path assumes R2=0 for every action). Set entropy_regularization=False "
                    "for bi reactions."
                )

    def _template_mask_info(self, smiles_batch):
        rm = self.env.unwrapped.reaction_manager
        mask_kind = td3_template_mask_kind(self.env, override=self._template_mask_kind_override)
        template_types = rm.template_types.to(device)
        masks = [rm.get_mask(smile, kind=mask_kind) for smile in smiles_batch]
        mask_tensor = torch.stack(masks).to(device)

        extra = self.template_dim - int(mask_tensor.shape[-1])
        if extra < 0:
            raise ValueError(
                f"TD3 policy template_dim={self.template_dim} is smaller than mask size={mask_tensor.shape[-1]}"
            )
        if extra:
            # Extra action slots are Stop/no-op slots. They are legal for Stop-enabled envs
            # and have template type 0 so the R2 head stays inactive.
            mask_tensor = torch.cat([mask_tensor, torch.ones(mask_tensor.shape[0], extra, device=device)], dim=-1)
            template_types = torch.cat([template_types, torch.zeros(extra, dtype=template_types.dtype, device=device)])
        return mask_tensor, template_types

    def get_action(self, state, evaluate=False):
        state = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        mask_info = None
        if hasattr(self.env.unwrapped, "reaction_manager") and hasattr(self.env.unwrapped, "current_state"):
            mask_info = self._template_mask_info([self.env.unwrapped.current_state])
        self.actor.eval() if evaluate else self.actor.train()

        # Entropy-regularized training-time sampling: draw a template from the
        # masked softmax distribution. Eval still uses the original argmax path
        # (so eval is a deterministic head-to-head against vanilla TD3).
        if self.entropy_regularization and not evaluate and mask_info is not None:
            with torch.no_grad():
                logits = self.actor.f_net(state)
                template_mask, _ = mask_info
                masked_logits = logits + (1.0 - template_mask) * (-1e9)
                probs = F.softmax(masked_logits, dim=-1)
                template_idx = torch.distributions.Categorical(probs=probs).sample()
                template_one_hot = F.one_hot(template_idx, num_classes=self.template_dim).float()
                r2_vector = state.new_zeros(state.size(0), self.continuous_r2_dim)
            return template_one_hot, r2_vector

        with torch.no_grad():
            template, r2_vector = self.actor(state, mask_info, self.temperature, evaluate=evaluate)
        if self.continuous_r2_dim > 0 and torch.any(r2_vector != 0) and not evaluate:
            r2_vector = r2_vector + torch.randn_like(r2_vector) * self.noise_std
        return template, r2_vector

    def _q_all_actions(self, critic_net, state_obs: torch.Tensor) -> torch.Tensor:
        """Return ``Q(s, e_i)`` for every template index ``i`` as ``(B, template_dim)``.

        Assumes R2 = 0 for every action, which holds for ``reaction_mode='uni'``
        (uni templates never use the R2 head). Used by the entropy-regularized
        update to compute ``E_{a~π}[Q(s,a)]`` without a separate Q-head.
        """
        batch_size = state_obs.size(0)
        n_actions = self.template_dim
        state_rep = state_obs.unsqueeze(1).expand(-1, n_actions, -1).reshape(batch_size * n_actions, -1)
        eye = torch.eye(n_actions, device=state_obs.device, dtype=state_obs.dtype)
        actions_rep = eye.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * n_actions, n_actions)
        if self.continuous_r2_dim > 0:
            r2_rep = state_obs.new_zeros(batch_size * n_actions, self.continuous_r2_dim)
        else:
            r2_rep = state_obs.new_zeros(batch_size * n_actions, 0)
        q_flat = critic_net(state_rep, actions_rep, r2_rep)
        return q_flat.reshape(batch_size, n_actions)

    def train(self, replay_buffer, batch_size=32):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        if self.entropy_regularization:
            return self._train_entropy(replay_buffer, batch_size)
        return self._train_deterministic(replay_buffer, batch_size)

    def _train_deterministic(self, replay_buffer, batch_size: int) -> dict:
        """Original TD3 update: deterministic-greedy actor, clipped target noise on R2."""
        self.total_it += 1

        (
            state_smiles,
            state_obs,
            templates,
            r2_vectors,
            rewards,
            next_state_smiles,
            next_state_obs,
            not_dones,
        ) = replay_buffer.sample(batch_size)

        masks_info = self._template_mask_info(state_smiles)
        next_masks_info = self._template_mask_info(next_state_smiles)

        with torch.no_grad():
            # Deterministic target actions + clipped Gaussian noise on the continuous R2 head only
            # (standard TD3). Stochastic Gumbel templates here inject biased, high-variance targets.
            next_templates, next_r2_vectors = self.actor_target(
                next_state_obs, next_masks_info, self.temperature, evaluate=True
            )
            noise = (torch.randn_like(next_r2_vectors) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            _, template_types_next = next_masks_info
            next_template_idx = next_templates.argmax(dim=-1)
            is_bimolecular = (
                (template_types_next[next_template_idx] == 1).reshape(-1, 1).to(dtype=noise.dtype)
                if self.continuous_r2_dim > 0
                else torch.zeros((next_state_obs.size(0), 1), device=noise.device, dtype=noise.dtype)
            )
            next_r2_vectors = (next_r2_vectors + noise * is_bimolecular).clamp(-self.max_action, self.max_action)
            target_q1 = self.critic1_target(next_state_obs, next_templates, next_r2_vectors)
            target_q2 = self.critic2_target(next_state_obs, next_templates, next_r2_vectors)
            target_q = rewards + not_dones * self.gamma * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(state_obs, templates, r2_vectors)
        current_q2 = self.critic2(state_obs, templates, r2_vectors)
        self.critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            current_templates, current_r2_vectors = self.actor(state_obs, masks_info, self.temperature)
            self.actor_loss = -self.critic1(state_obs, current_templates, current_r2_vectors).mean()
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()
            self._update_target(self.critic1, self.critic1_target, self.tau)
            self._update_target(self.critic2, self.critic2_target, self.tau)
            self._update_target(self.actor, self.actor_target, self.tau)

        self.temperature = max(self.temperature_end, self.temperature * self.temperature_decay)

        return {
            "total_iterations": self.total_it,
            "critic_loss": self.critic_loss.item(),
            "actor_loss": self.actor_loss if self.actor_loss is None else self.actor_loss.item(),
            "current_q_values": current_q1.mean().item(),
            "target_q_values": target_q.mean().item(),
            "temperature": self.temperature,
            "entropy_regularization": 0.0,
        }

    def _train_entropy(self, replay_buffer, batch_size: int) -> dict:
        """SAC-discrete-style soft actor-critic update.

        Differences from the standard TD3 update:
          • Actor as a masked-softmax categorical; trained to maximize
            ``E_{a~π}[Q(s,a) - α log π(a|s)]`` (entropy bonus).
          • Critic target is the soft Bellman backup
            ``y = r + γ * E_{a~π_target}[min(Q1', Q2') - α log π_target(a|s')]``.
          • Critics are still trained via MSE on the actually-sampled actions
            from the replay buffer.

        Restricted to uni reactions (R2 assumed zero for every action). Other
        algorithms (PPO/A2C/GraphTransRL) are unaffected by this code path.
        """
        self.total_it += 1

        (
            state_smiles,
            state_obs,
            templates,
            r2_vectors,
            rewards,
            next_state_smiles,
            next_state_obs,
            not_dones,
        ) = replay_buffer.sample(batch_size)

        template_mask, _ = self._template_mask_info(state_smiles)
        next_template_mask, _ = self._template_mask_info(next_state_smiles)

        with torch.no_grad():
            next_logits = self.actor_target.f_net(next_state_obs)
            next_masked_logits = next_logits + (1.0 - next_template_mask) * (-1e9)
            next_log_probs = F.log_softmax(next_masked_logits, dim=-1)
            next_probs = next_log_probs.exp()
            # Replace -inf log_probs (masked-out actions) with 0 so they
            # contribute 0 to the expectation (probs are also 0 there).
            # Avoids NaN from 0 * (-inf).
            next_log_probs_safe = torch.where(
                next_template_mask.bool(), next_log_probs, torch.zeros_like(next_log_probs)
            )
            q1_next_all = self._q_all_actions(self.critic1_target, next_state_obs)
            q2_next_all = self._q_all_actions(self.critic2_target, next_state_obs)
            q_min_next = torch.min(q1_next_all, q2_next_all)
            soft_v_next = (
                next_probs * (q_min_next - self.entropy_alpha * next_log_probs_safe)
            ).sum(dim=-1, keepdim=True)
            target_q = rewards + not_dones * self.gamma * soft_v_next

        current_q1 = self.critic1(state_obs, templates, r2_vectors)
        current_q2 = self.critic2(state_obs, templates, r2_vectors)
        self.critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        policy_entropy_value = 0.0
        if self.total_it % self.policy_freq == 0:
            logits = self.actor.f_net(state_obs)
            masked_logits = logits + (1.0 - template_mask) * (-1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = log_probs.exp()
            log_probs_safe = torch.where(
                template_mask.bool(), log_probs, torch.zeros_like(log_probs)
            )
            q1_all = self._q_all_actions(self.critic1, state_obs)
            actor_objective = (probs * (q1_all - self.entropy_alpha * log_probs_safe)).sum(dim=-1)
            self.actor_loss = -actor_objective.mean()
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()
            self._update_target(self.critic1, self.critic1_target, self.tau)
            self._update_target(self.critic2, self.critic2_target, self.tau)
            self._update_target(self.actor, self.actor_target, self.tau)

            with torch.no_grad():
                policy_entropy_value = float(
                    -(probs * log_probs_safe).sum(dim=-1).mean().item()
                )

        self.temperature = max(self.temperature_end, self.temperature * self.temperature_decay)

        return {
            "total_iterations": self.total_it,
            "critic_loss": self.critic_loss.item(),
            "actor_loss": self.actor_loss if self.actor_loss is None else self.actor_loss.item(),
            "current_q_values": current_q1.mean().item(),
            "target_q_values": target_q.mean().item(),
            "temperature": self.temperature,
            "entropy_regularization": 1.0,
            "entropy_alpha": self.entropy_alpha,
            "policy_entropy": policy_entropy_value,
        }

    def _update_target(self, source, target, tau):
        for target_param, param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, filename, steps_done, episode_count, replay_buffer, *, include_replay_buffer: bool = False):
        state_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "temperature": self.temperature,
            "actor_loss": _loss_value_for_checkpoint(self.actor_loss),
            "critic_loss": _loss_value_for_checkpoint(self.critic_loss),
            "steps_done": steps_done,
            "episode_count": episode_count,
        }
        if include_replay_buffer:
            state_dict["replay_buffer"] = replay_buffer
        torch.save(state_dict, filename, pickle_protocol=5)

    def apply_checkpoint(self, checkpoint, source_label="checkpoint"):
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_it = checkpoint.get("total_it", 0)
        self.temperature = checkpoint.get("temperature", 1.0)
        self.actor_loss = checkpoint.get("actor_loss", None)
        self.critic_loss = checkpoint.get("critic_loss", None)
        return checkpoint.get("steps_done", 0), checkpoint.get("episode_count", 0), checkpoint.get("replay_buffer")

    def load_model(self, filename):
        try:
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location="cpu")
        return self.apply_checkpoint(checkpoint, source_label=filename)


__all__ = ["TD3Agent"]
