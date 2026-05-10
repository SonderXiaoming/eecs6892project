# envs.py

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor

from wrappers import (
    ScaledActionBonusWrapper,
    StateVisitBonusWrapper,
    PotentialBasedShapingWrapper,
    RewardShapingWrapper,
    HeatmapWrapper,
)


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
        # Same (s,a) counting as MiniGrid ActionBonus, but scaled so dense bonus does not
        # overwhelm sparse task reward (official wrapper uses ~1.0/step on new pairs).
        env = ScaledActionBonusWrapper(env, bonus_scale=0.05)

    elif strategy == "state_bonus":
        env = StateVisitBonusWrapper(env, bonus_coef=0.05)

    elif strategy == "reward_shaping":
        env = RewardShapingWrapper(env, step_penalty=-0.001, success_bonus=0.0)

    elif strategy == "potential_shaping":
        # Match PPO gamma in train.py for policy-invariant shaping.
        env = PotentialBasedShapingWrapper(env, gamma=0.99, scale=1.0)

    elif strategy in ["baseline", "high_entropy"]:
        pass

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if record_heatmap:
        env = HeatmapWrapper(env)

    env = Monitor(env)
    env.reset(seed=seed)

    return env
