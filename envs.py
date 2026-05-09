# envs.py

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper, ActionBonus
from stable_baselines3.common.monitor import Monitor

from wrappers import StateVisitBonusWrapper, RewardShapingWrapper, HeatmapWrapper


def make_env(env_id: str, strategy: str, seed: int, record_heatmap: bool = False):
    """
    Create a MiniGrid environment with selected exploration strategy.

    Important implementation detail:
    MiniGrid image observations are usually 7x7x3. Stable-Baselines3's default
    CnnPolicy uses an 8x8 first convolution kernel, which crashes on 7x7 inputs.
    Therefore this project uses ImgObsWrapper + FlattenObservation + MlpPolicy.
    """

    env = gym.make(env_id, render_mode="rgb_array")

    # Keep only the compact MiniGrid image observation: usually 7 x 7 x 3.
    env = ImgObsWrapper(env)

    # Convert 7x7x3 image observation to a flat vector so PPO can use MlpPolicy.
    env = FlattenObservation(env)

    if strategy == "action_bonus":
        # MiniGrid official exploration bonus wrapper.
        env = ActionBonus(env)

    elif strategy == "state_bonus":
        env = StateVisitBonusWrapper(env, bonus_coef=0.05)

    elif strategy == "reward_shaping":
        env = RewardShapingWrapper(env, step_penalty=-0.001, success_bonus=0.0)

    elif strategy in ["baseline", "high_entropy"]:
        pass

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if record_heatmap:
        env = HeatmapWrapper(env)

    env = Monitor(env)
    env.reset(seed=seed)

    return env
