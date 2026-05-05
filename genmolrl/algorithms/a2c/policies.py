"""A2C policies with optional observation-tail action masking."""

from __future__ import annotations

import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution, MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Mask Discrete or MultiDiscrete logits using the observation tail."""

    def _masked_action_dist(self, obs: th.Tensor, latent_pi: th.Tensor):
        logits = self.action_net(latent_pi)
        if not isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution)):
            raise ValueError("MaskedActorCriticPolicy supports Discrete and MultiDiscrete action spaces.")
        mask = obs[..., -logits.shape[-1] :] > 0.5
        no_valid = ~mask.any(dim=-1)
        if no_valid.any():
            mask = mask.clone()
            mask[no_valid, -1] = True
        masked_logits = logits.masked_fill(~mask, -1.0e8)
        return self.action_dist.proba_distribution(action_logits=masked_logits)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._masked_action_dist(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def get_distribution(self, obs: th.Tensor):
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._masked_action_dist(obs, latent_pi)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._masked_action_dist(obs, latent_pi)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()
