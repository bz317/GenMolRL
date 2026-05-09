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
        auto_tune_alpha: bool = False,
        target_entropy: float = 0.5,
        target_entropy_ratio: float | None = None,
        alpha_lr: float = 3e-4,
    ):
        self.env = env
        # ``state_dim`` is the size of what the actor / critic *see* as an
        # observation. With ``append_action_mask_to_obs=True`` (now the
        # default for TD3, matching PPO/A2C) this is morgan_fp + action_mask.
        self.state_dim = env.unwrapped.observation_space.shape[0]
        self.template_dim = env.unwrapped.action_space.n
        # ``fingerprint_dim`` is the morgan-FP width itself (1024). It is used
        # for the continuous R2 head (which outputs a fingerprint-shaped vector
        # of the second reactant) and is decoupled from ``state_dim`` so that
        # appending the action mask to the observation does NOT inflate the
        # R2 vector. Falls back to obs dim for any env that doesn't expose
        # ``base_obs_dim`` (back-compat).
        fingerprint_dim = int(getattr(env.unwrapped, "base_obs_dim", self.state_dim))
        self.discrete_uni = getattr(env.unwrapped, "action_design", "") == TD3_UNI_DISCRETE_ACTION_DESIGN
        self.continuous_r2_dim = 0 if self.discrete_uni else fingerprint_dim

        if self.discrete_uni:
            self.actor = ActorNetworkUniDiscrete(self.state_dim, self.template_dim).to(device)
        else:
            self.actor = ActorNetwork(self.state_dim, self.template_dim, fingerprint_dim).to(device)
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

        # Optional automatic alpha tuning (SAC-discrete style). When enabled,
        # ``alpha`` becomes a learnable parameter that adjusts every actor
        # update to keep the policy entropy near ``target_entropy``. Initial
        # alpha = ``entropy_alpha``. ``alpha`` always stays positive via the
        # log-parameterization. When ``auto_tune_alpha=False`` the agent uses
        # the fixed ``entropy_alpha`` (original behavior).
        #
        # Two ways to specify the target entropy:
        #   • ``target_entropy_ratio`` (preferred when set): a per-state target
        #     ``H_target(s) = ratio * log(N_feasible(s))`` so the constraint
        #     scales with how many actions the state actually offers. The
        #     batch mean of these per-state targets is used in alpha_loss.
        #     This avoids over-regularizing binary-choice states (where
        #     log(2)=0.69 means a fixed-nats target near that value would
        #     force the policy to stay near uniform on Stop-vs-react states).
        #   • ``target_entropy`` (fallback when ratio is None): a single
        #     scalar in nats applied as the global batch-mean target.
        self.auto_tune_alpha = bool(auto_tune_alpha) and self.entropy_regularization
        self.target_entropy = float(target_entropy)
        self.target_entropy_ratio = (
            None if target_entropy_ratio is None else float(target_entropy_ratio)
        )
        if self.auto_tune_alpha:
            initial_log_alpha = math.log(max(self.entropy_alpha, 1e-8))
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(initial_log_alpha, dtype=torch.float32, device=device)
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(alpha_lr))
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

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

        # ``alpha`` is either the fixed config value or a learnable parameter
        # under auto-tuning. Treat it as a scalar tensor either way.
        if self.auto_tune_alpha:
            alpha = self.log_alpha.detach().exp()
        else:
            alpha = torch.tensor(self.entropy_alpha, device=state_obs.device, dtype=state_obs.dtype)

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
                next_probs * (q_min_next - alpha * next_log_probs_safe)
            ).sum(dim=-1, keepdim=True)
            target_q = rewards + not_dones * self.gamma * soft_v_next

        current_q1 = self.critic1(state_obs, templates, r2_vectors)
        current_q2 = self.critic2(state_obs, templates, r2_vectors)
        self.critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        policy_entropy_value = 0.0
        alpha_loss_value = 0.0
        # Default to fixed-nats target on non-actor-update steps. Overwritten
        # below on actor-update steps when ``target_entropy_ratio`` is active.
        effective_target_entropy = self.target_entropy
        if self.total_it % self.policy_freq == 0:
            logits = self.actor.f_net(state_obs)
            masked_logits = logits + (1.0 - template_mask) * (-1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = log_probs.exp()
            log_probs_safe = torch.where(
                template_mask.bool(), log_probs, torch.zeros_like(log_probs)
            )
            q1_all = self._q_all_actions(self.critic1, state_obs)
            actor_objective = (probs * (q1_all - alpha * log_probs_safe)).sum(dim=-1)
            self.actor_loss = -actor_objective.mean()
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()
            self._update_target(self.critic1, self.critic1_target, self.tau)
            self._update_target(self.critic2, self.critic2_target, self.tau)
            self._update_target(self.actor, self.actor_target, self.tau)

            with torch.no_grad():
                entropy_per_state = -(probs * log_probs_safe).sum(dim=-1)
                policy_entropy_value = float(entropy_per_state.mean().item())

            # SAC-discrete alpha auto-tune: minimize ``alpha * (H - H_target)``
            # so ∂L/∂log_alpha = α * (H_actual - H_target). When entropy is
            # below target, gradient is negative and log_alpha increases (more
            # entropy regularization). When above target, log_alpha decreases.
            if self.auto_tune_alpha:
                # Per-state target = ratio * log(N_feasible(s)) when the ratio
                # knob is set; otherwise fall back to the fixed-nats target.
                # The per-state form scales the entropy budget with how many
                # actions each state actually has — important here because
                # ~40% of uni starts have only 2 feasible actions (binary
                # Stop-vs-react), where a fixed 0.6-nats target is too high.
                if self.target_entropy_ratio is not None:
                    n_feasible = template_mask.sum(dim=-1).clamp(min=1.0)
                    target_per_state = self.target_entropy_ratio * n_feasible.log()
                    target_for_loss = target_per_state.mean().detach()
                else:
                    target_for_loss = torch.tensor(
                        self.target_entropy,
                        device=entropy_per_state.device,
                        dtype=entropy_per_state.dtype,
                    )
                alpha_for_loss = self.log_alpha.exp()
                alpha_loss = (
                    alpha_for_loss
                    * (entropy_per_state.detach().mean() - target_for_loss)
                )
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_value = float(alpha_loss.detach().item())
                effective_target_entropy = float(target_for_loss.item())
            else:
                effective_target_entropy = self.target_entropy

        self.temperature = max(self.temperature_end, self.temperature * self.temperature_decay)

        current_alpha = (
            float(self.log_alpha.detach().exp().item())
            if self.auto_tune_alpha
            else self.entropy_alpha
        )
        return {
            "total_iterations": self.total_it,
            "critic_loss": self.critic_loss.item(),
            "actor_loss": self.actor_loss if self.actor_loss is None else self.actor_loss.item(),
            "current_q_values": current_q1.mean().item(),
            "target_q_values": target_q.mean().item(),
            "temperature": self.temperature,
            "entropy_regularization": 1.0,
            "entropy_alpha": current_alpha,
            "auto_tune_alpha": 1.0 if self.auto_tune_alpha else 0.0,
            "target_entropy": effective_target_entropy,
            "target_entropy_ratio": (
                self.target_entropy_ratio if self.target_entropy_ratio is not None else 0.0
            ),
            "alpha_loss": alpha_loss_value,
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
        if self.auto_tune_alpha:
            state_dict["log_alpha"] = float(self.log_alpha.detach().item())
            state_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
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
        if self.auto_tune_alpha and "log_alpha" in checkpoint:
            with torch.no_grad():
                self.log_alpha.copy_(
                    torch.tensor(float(checkpoint["log_alpha"]), device=self.log_alpha.device, dtype=self.log_alpha.dtype)
                )
            if "alpha_optimizer_state_dict" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        return checkpoint.get("steps_done", 0), checkpoint.get("episode_count", 0), checkpoint.get("replay_buffer")

    def load_model(self, filename):
        try:
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location="cpu")
        return self.apply_checkpoint(checkpoint, source_label=filename)


__all__ = ["TD3Agent"]
