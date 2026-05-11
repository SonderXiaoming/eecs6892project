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

    def __init__(self, env, bonus_coef: float = 0.005):
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
    Potential-based reward shaping:
        F(s, s') = gamma * Phi(s') - Phi(s)

    Phi(s) is defined as negative normalized shortest-path distance
    from the agent to the current subgoal.

    Compared with Manhattan distance, shortest-path distance respects
    walls and room topology, which is important in FourRooms-like maps.
    """

    def __init__(self, env, gamma: float = 0.99, scale: float = 0.05):
        super().__init__(env)
        self.gamma = float(gamma)
        self.scale = float(scale)
        self._phi_prev: float = 0.0
        


    def _scan_grid(self, e):
        from minigrid.core.world_object import Key, Door, Goal

        key_pos = None
        door_pos = None
        goal_pos = None
        door_open = True

        for y in range(e.height):
            for x in range(e.width):
                obj = e.grid.get(x, y)

                if obj is None:
                    continue

                if isinstance(obj, Key):
                    key_pos = (x, y)

                elif isinstance(obj, Door):
                    door_pos = (x, y)
                    door_open = bool(obj.is_open)

                elif isinstance(obj, Goal):
                    goal_pos = (x, y)

        return key_pos, door_pos, goal_pos, door_open

    def _subgoal(self, e):
        from minigrid.core.world_object import Key

        key_pos, door_pos, goal_pos, door_open = self._scan_grid(e)

        carrying = getattr(e, "carrying", None)
        has_key = isinstance(carrying, Key)

        # Stage 1: if there is a key and the agent is not carrying it, go to key.
        if not has_key and key_pos is not None:
            return key_pos

        # Stage 2: if the agent has key and door is still closed, go to door.
        if has_key and door_pos is not None and not door_open:
            return door_pos

        # Stage 3: otherwise go to goal.
        if goal_pos is not None:
            return goal_pos

        # Fallback: no meaningful subgoal found.
        return tuple(e.agent_pos)

    def _is_passable(self, e, x: int, y: int, target: tuple[int, int]) -> bool:
        """
        Return whether the BFS search can pass through cell (x, y).

        Walls are not passable.
        Empty cells are passable.
        Goal/key cells are passable.
        Open doors are passable.
        Closed doors are passable only if the door cell is the current target.
        """
        obj = e.grid.get(x, y)

        if obj is None:
            return True

        # Always allow reaching the current target cell.
        if (x, y) == target:
            return True

        obj_type = getattr(obj, "type", None)

        if obj_type in ["goal", "key"]:
            return True

        if obj_type == "door":
            return bool(getattr(obj, "is_open", False))

        return False

    def _shortest_path_distance(self, e, start: tuple[int, int], target: tuple[int, int]) -> float:
        """
        Compute shortest-path distance from start to target using BFS.

        This respects walls and closed doors, unlike Manhattan distance.
        """
        from collections import deque
        import math

        if start == target:
            return 0.0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), dist = queue.popleft()

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx = x + dx
                ny = y + dy
                nxt = (nx, ny)

                if nx < 0 or nx >= e.width or ny < 0 or ny >= e.height:
                    continue

                if nxt in visited:
                    continue

                if not self._is_passable(e, nx, ny, target):
                    continue

                if nxt == target:
                    return float(dist + 1)

                visited.add(nxt)
                queue.append((nxt, dist + 1))

        return math.inf

    def _potential(self) -> float:
        import math

        e = self.env.unwrapped

        pos = tuple(e.agent_pos)
        target = self._subgoal(e)

        dist = self._shortest_path_distance(e, pos, target)

        # If target is unreachable under the current passability rule,
        # return a conservative low potential.
        if math.isinf(dist):
            return -1.0

        # Normalize by map area to keep shaping reward small.
        denom = float(max(e.width * e.height, 1))
        phi = -dist / denom

        return float(phi)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi_prev = self._potential()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        phi = self._potential()

        env_reward = float(info.get("env_reward", reward))

        # If the environment already gives a successful terminal reward,
        # avoid modifying that final signal with shaping.
        if terminated and env_reward > 0:
            shaping = 0.0
        else:
            shaping = self.gamma * phi - self._phi_prev

        self._phi_prev = phi

        total = float(reward) + self.scale * shaping

        info["env_reward"] = env_reward
        info["potential_phi"] = phi
        info["shaping"] = shaping
        info["total_reward"] = total
        info["subgoal"] = self._subgoal(self.env.unwrapped)

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
