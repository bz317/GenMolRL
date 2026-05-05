"""Unified molecule-design environment for PPO, A2C, and TD3/PGFS."""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from genmolrl.chem.datasets import load_pickle
from genmolrl.chem.fingerprints import morgan_fp_array
from genmolrl.chem.reaction_manager import BI_TYPE, ReactionManager
from genmolrl.envs.action_spaces import ActionSpaceSpec
from genmolrl.envs.masking import MaskProvider
from genmolrl.envs.rewards import RewardFunction, qed
from genmolrl.envs.start_strategies import StartStrategy

logger = logging.getLogger(__name__)


class MoleculeDesignEnv(gym.Env):
    """Configurable reaction-template environment.

    `sb3_multidiscrete` intentionally preserves the current factorized
    `MultiDiscrete([T, R2])` design for Bi mode.
    """

    metadata = {"render_modes": ["human", "console", "rgb_array"]}

    def __init__(
        self,
        reactant_file: str,
        template_file: str,
        reaction_mode: str = "uni",
        algorithm_family: str = "sb3_discrete",
        action_design: str = "discrete",
        masking: str = "substructure",
        reward: str = "delta_qed",
        max_steps: int = 5,
        render_mode: str | None = None,
        start_strategy: str = "random_pool",
        start_smiles_file: str | None = None,
        fixed_start_smiles: str | None = None,
        use_stop_action: bool = True,
        stop_early_penalty: float = 0.0,
        stop_penalty_until_step: int = -1,
        invalid_reaction_penalty: float = -1.0,
        reward_round_digits: int | None = None,
        info_qed_round_digits: int | None = None,
        append_action_mask_to_obs: bool | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.current_step = 0
        self.current_state: str | None = None
        self.previous_state: str | None = None
        self.initial_qed = 0.0
        self.steps_log: dict[int, dict] = {}
        self.reaction_mode = reaction_mode
        self.algorithm_family = algorithm_family
        self.action_design = action_design
        self.use_stop_action = bool(use_stop_action)
        self.stop_early_penalty = float(stop_early_penalty)
        self.stop_penalty_until_step = int(stop_penalty_until_step)

        self.reactants = load_pickle(Path(reactant_file))
        raw_templates = load_pickle(Path(template_file))
        all_reactions = ReactionManager(raw_templates, self.reactants)
        self.templates = all_reactions.templates_for_mode(reaction_mode)
        self.reaction_manager = ReactionManager(self.templates, self.reactants)
        self.num_templates = len(self.templates)
        self.reactant_keys = list(self.reactants.keys())

        self.mask_provider = MaskProvider(masking, use_stop_action=self.use_stop_action)
        self.reward_fn = RewardFunction(
            reward,
            invalid_penalty=invalid_reaction_penalty,
            round_digits=reward_round_digits,
        )
        self.info_qed_round_digits = info_qed_round_digits
        self.start_strategy = StartStrategy(start_strategy, fixed_start_smiles, start_smiles_file)
        self.start_strategy.initialize(self.reactants)

        spec = ActionSpaceSpec(
            family=algorithm_family,
            reaction_mode=reaction_mode,
            action_design=action_design,
            use_stop_action=self.use_stop_action,
        )
        self.action_space = spec.build(self.num_templates, len(self.reactants))
        if append_action_mask_to_obs is None:
            append_action_mask_to_obs = algorithm_family in {"sb3_discrete", "sb3_multidiscrete"}
        self.append_action_mask_to_obs = bool(append_action_mask_to_obs)
        self.base_obs_dim = 1024
        mask_dim = len(self.action_masks()) if self.append_action_mask_to_obs else 0
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.base_obs_dim + mask_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.steps_log = {}
        self.current_state = self.start_strategy.sample(self)
        if not self.start_strategy.validate(self.current_state):
            raise ValueError(f"Invalid start molecule: {self.current_state}")
        self.previous_state = self.current_state
        self.initial_qed = qed(self.current_state)
        return self._get_obs(), self._get_info()

    def action_masks(self) -> np.ndarray:
        if self.algorithm_family == "sb3_multidiscrete":
            return self.mask_provider.multidiscrete_mask(self.reaction_manager, self.current_state)
        return self.mask_provider.template_mask_with_stop(self.reaction_manager, self.current_state)

    def _get_obs(self) -> np.ndarray:
        obs = morgan_fp_array(self.current_state)
        if not self.append_action_mask_to_obs:
            return obs
        mask = self.action_masks().astype(np.float32)
        return np.concatenate([obs, mask]).astype(np.float32, copy=False)

    def _get_info(self) -> dict:
        q = qed(self.current_state)
        if self.info_qed_round_digits is not None:
            q = round(q, int(self.info_qed_round_digits))
        return {
            "SMILES": self.current_state,
            "QED": q,
            "initial_QED": self.initial_qed,
            "step": self.current_step,
        }

    def _parse_action(self, action) -> tuple[int, str | None]:
        if self.algorithm_family == "td3_pgfs":
            if isinstance(action, tuple):
                template = action[0]
                if hasattr(template, "detach"):
                    template = int(template.detach().reshape(-1).argmax().item())
                r2 = action[1]
                if hasattr(r2, "detach"):
                    return int(template), None
                return int(template), r2
            return int(action), None
        if self.algorithm_family == "sb3_multidiscrete":
            template_index = int(action[0])
            reactant_index = int(action[1])
            if template_index >= self.num_templates:
                return template_index, None
            template = self.templates[template_index]
            if template.get("type") == BI_TYPE:
                return template_index, self.reactant_keys[reactant_index]
            return template_index, None
        return int(action), None

    def step(self, action):
        self.current_step += 1
        template_index, r2 = self._parse_action(action)

        if self.use_stop_action and template_index == self.num_templates:
            reward = self.reward_fn.stop_reward(
                current_step=self.current_step,
                stop_early_penalty=self.stop_early_penalty,
                stop_penalty_until_step=self.stop_penalty_until_step,
            )
            info = self._get_info()
            info.update({"stop": True, "stop_reward": reward})
            return self._get_obs(), reward, False, True, info

        template = self.templates.get(template_index)
        if template is None:
            info = self._get_info()
            info["bad_template_index"] = template_index
            return self._get_obs(), self.reward_fn.invalid_penalty, False, True, info

        previous_state = self.current_state
        new_state = self.reaction_manager.apply_reaction(previous_state, template, r2)
        if new_state is None:
            self.current_state = None
            info = self._get_info()
            info["reaction_failed"] = True
            return np.zeros(self.observation_space.shape[0], dtype=np.float32), self.reward_fn.invalid_penalty, False, True, info

        self.steps_log[self.current_step] = {
            "r1": previous_state,
            "template": template.get("name", str(template_index)),
            "r2": r2,
            "product": new_state,
        }
        self.previous_state = previous_state
        self.current_state = new_state
        reward = self.reward_fn.step_reward(previous_state, new_state)
        terminated = self.current_step >= self.max_steps
        has_next_template = bool(
            self.reaction_manager.feasible_first_reactant_templates(
                new_state,
                kind=self.mask_provider.mode,
            )
        )
        truncated = not has_next_template
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "console":
            for step, item in self.steps_log.items():
                print(f"Step {step}: {item}")
        return None

    def close(self):
        return None
