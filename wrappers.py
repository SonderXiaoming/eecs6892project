# wrappers.py

import math
import gymnasium as gym
import numpy as np
from collections import defaultdict


def _int_action(action) -> int:
    """VecEnv / numpy-safe discrete action for hashing and env.step."""
    if isinstance(action, (int, np.integer)):
        return int(action)
    return int(np.asarray(action).item())


class ScaledActionBonusWrapper(gym.Wrapper):
    """
    Same idea as minigrid.wrappers.ActionBonus (count-based bonus on (pos, dir, action)),
    but with a tunable scale.

    MiniGrid's ActionBonus uses 1/sqrt(n) (~1.0 on first visits). With sparse env reward,
    that dense signal dominates PPO's objective and the policy often never learns to finish
    the task (near-zero success under raw reward eval).
    """

    def __init__(self, env, bonus_scale: float = 0.05):
        super().__init__(env)
        self.bonus_scale = float(bonus_scale)
        self.counts: dict[tuple, int] = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env = self.unwrapped
        a = _int_action(action)
        tup = (tuple(env.agent_pos), int(env.agent_dir), a)

        pre = self.counts.get(tup, 0)
        new_count = pre + 1
        self.counts[tup] = new_count

        bonus = self.bonus_scale / math.sqrt(new_count)
        total = float(reward) + bonus
        info["env_reward"] = reward
        info["intrinsic_bonus"] = bonus
        info["total_reward"] = total
        return obs, total, terminated, truncated, info


class StateVisitBonusWrapper(gym.Wrapper):
    """
    Count-based intrinsic reward on coarse grid state (not raw pixels):
        r_total = r_env + beta / sqrt(N(s))

    Hashing the full flattened image makes N(s) grow slowly for "real" revisits and can
    add a large almost-per-step bonus (many unique-looking obs), which again drowns out
    sparse success signal. Using (agent_pos, agent_dir, carried_type) matches the
    underlying tabular structure of MiniGrid.
    """

    def __init__(self, env, bonus_coef: float = 0.05):
        super().__init__(env)
        self.bonus_coef = bonus_coef
        self.visit_counts = defaultdict(int)

    def _state_key(self):
        e = self.env.unwrapped
        carrying = getattr(e, "carrying", None)
        carry_tag = type(carrying).__name__ if carrying is not None else None
        return (tuple(e.agent_pos), int(e.agent_dir), carry_tag)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        key = self._state_key()
        self.visit_counts[key] += 1

        intrinsic_bonus = self.bonus_coef / np.sqrt(self.visit_counts[key])
        total_reward = float(reward) + intrinsic_bonus

        info["env_reward"] = reward
        info["intrinsic_bonus"] = intrinsic_bonus
        info["total_reward"] = total_reward

        return obs, total_reward, terminated, truncated, info


class PotentialBasedShapingWrapper(gym.Wrapper):
    """
    Potential-based shaping (Ng et al., ICML 1999): F(s,s') = γ Φ(s') − Φ(s).

    Φ is negative normalized Manhattan distance to the current subgoal:
    key still on grid → navigate to key; carrying key and door closed → door;
    otherwise → goal. Environments without keys/doors fall back to goal-only.

    This preserves the optimal policy in tabular settings and gives dense feedback
    aligned with task structure (often stronger than generic count-based bonuses).
    """

    def __init__(self, env, gamma: float = 0.99, scale: float = 1.0):
        super().__init__(env)
        self.gamma = float(gamma)
        self.scale = float(scale)
        self._phi_prev: float = 0.0

    @staticmethod
    def _manhattan(a, b) -> int:
        return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _scan_grid(self, e):
        from minigrid.core.world_object import Key, Door, Goal

        key_pos = door_pos = goal_pos = None
        door_open = True
        for j in range(e.height):
            for i in range(e.width):
                c = e.grid.get(i, j)
                if c is None:
                    continue
                if isinstance(c, Key):
                    key_pos = (i, j)
                if isinstance(c, Door):
                    door_pos = (i, j)
                    door_open = bool(c.is_open)
                if isinstance(c, Goal):
                    goal_pos = (i, j)
        return key_pos, door_pos, goal_pos, door_open

    def _subgoal(self, e):
        from minigrid.core.world_object import Key

        key_pos, door_pos, goal_pos, door_open = self._scan_grid(e)
        carrying = getattr(e, "carrying", None)
        has_key = isinstance(carrying, Key)

        if key_pos is not None:
            return key_pos
        if has_key and door_pos is not None and not door_open:
            return door_pos
        if goal_pos is not None:
            return goal_pos
        return tuple(e.agent_pos)

    def _potential(self) -> float:
        e = self.env.unwrapped
        pos = tuple(e.agent_pos)
        target = self._subgoal(e)
        dist = self._manhattan(pos, target)
        denom = float(max(e.width + e.height, 2))
        phi = -dist / denom
        return float(phi)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi_prev = self._potential()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        phi = self._potential()
        shaping = self.gamma * phi - self._phi_prev
        self._phi_prev = phi

        total = float(reward) + self.scale * shaping
        info["env_reward"] = reward
        info["potential_phi"] = phi
        info["shaping"] = shaping
        info["total_reward"] = total

        return obs, total, terminated, truncated, info


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
