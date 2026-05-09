# wrappers.py

import gymnasium as gym
import numpy as np
from collections import defaultdict


class StateVisitBonusWrapper(gym.Wrapper):
    """
    Count-based intrinsic reward:
        r_total = r_env + beta / sqrt(N(s))

    The observation is flattened into a 1D vector, so we convert it into bytes
    as a hashable state key.
    """

    def __init__(self, env, bonus_coef: float = 0.05):
        super().__init__(env)
        self.bonus_coef = bonus_coef
        self.visit_counts = defaultdict(int)

    def _state_key(self, obs):
        return np.asarray(obs).tobytes()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        key = self._state_key(obs)
        self.visit_counts[key] += 1

        intrinsic_bonus = self.bonus_coef / np.sqrt(self.visit_counts[key])
        total_reward = reward + intrinsic_bonus

        info["env_reward"] = reward
        info["intrinsic_bonus"] = intrinsic_bonus
        info["total_reward"] = total_reward

        return obs, total_reward, terminated, truncated, info


class RewardShapingWrapper(gym.Wrapper):
    """
    Simple reward shaping:
        - small step penalty to encourage shorter solutions
        - optional success bonus
    """

    def __init__(self, env, step_penalty: float = -0.001, success_bonus: float = 0.0):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.success_bonus = success_bonus

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = reward + self.step_penalty

        if terminated and reward > 0:
            shaped_reward += self.success_bonus

        info["env_reward"] = reward
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info


class HeatmapWrapper(gym.Wrapper):
    """
    Records visited agent positions for exploration heatmap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.position_counts = defaultdict(int)

    def _get_base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def _record_position(self):
        base_env = self._get_base_env()
        if hasattr(base_env, "agent_pos"):
            pos = tuple(base_env.agent_pos)
            self.position_counts[pos] += 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._record_position()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record_position()
        info["position_counts"] = dict(self.position_counts)
        return obs, reward, terminated, truncated, info
