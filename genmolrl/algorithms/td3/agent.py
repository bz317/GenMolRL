"""Self-contained PGFS-style TD3 agent."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim

from genmolrl.algorithms.td3.models import ActorNetwork, CriticNetwork

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
    ):
        self.env = env
        self.state_dim = env.unwrapped.observation_space.shape[0]
        self.template_dim = env.unwrapped.action_space.n
        self.action_dim = env.unwrapped.observation_space.shape[0]

        self.actor = ActorNetwork(self.state_dim, self.template_dim, self.action_dim).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1 = CriticNetwork(self.state_dim, self.template_dim, self.action_dim).to(device)
        self.critic2 = CriticNetwork(self.state_dim, self.template_dim, self.action_dim).to(device)
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

    def _template_mask_info(self, smiles_batch):
        template_types = self.env.unwrapped.reaction_manager.template_types.to(device)
        masks = [self.env.unwrapped.reaction_manager.get_feasible_mask(smile) for smile in smiles_batch]
        return torch.stack(masks).to(device), template_types

    def get_action(self, state, evaluate=False):
        state = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        mask_info = None
        if hasattr(self.env.unwrapped, "reaction_manager") and hasattr(self.env.unwrapped, "current_state"):
            mask_info = self._template_mask_info([self.env.unwrapped.current_state])
        self.actor.eval() if evaluate else self.actor.train()
        with torch.no_grad():
            template, r2_vector = self.actor(state, mask_info, self.temperature, evaluate=evaluate)
        if torch.any(r2_vector != 0) and not evaluate:
            r2_vector = r2_vector + torch.randn_like(r2_vector) * self.noise_std
        return template, r2_vector

    def train(self, replay_buffer, batch_size=32):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
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
            noise = (torch.randn_like(r2_vectors) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_templates, next_r2_vectors = self.actor_target(next_state_obs, next_masks_info, self.temperature)
            next_r2_vectors = (next_r2_vectors + noise).clamp(-self.max_action, self.max_action)
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

        return {
            "total_iterations": self.total_it,
            "critic_loss": self.critic_loss.item(),
            "actor_loss": self.actor_loss if self.actor_loss is None else self.actor_loss.item(),
            "current_q_values": current_q1.mean().item(),
            "target_q_values": target_q.mean().item(),
            "temperature": self.temperature,
        }

    def _update_target(self, source, target, tau):
        for target_param, param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, filename, steps_done, episode_count, replay_buffer):
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
            "replay_buffer": replay_buffer,
        }
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
        return checkpoint.get("steps_done", 0), checkpoint.get("episode_count", 0), checkpoint["replay_buffer"]

    def load_model(self, filename):
        try:
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location="cpu")
        return self.apply_checkpoint(checkpoint, source_label=filename)


__all__ = ["TD3Agent"]
