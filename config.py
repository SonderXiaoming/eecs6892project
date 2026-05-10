# config.py

import os

ENV_CONFIGS = {
    "FourRooms": "MiniGrid-FourRooms-v0",
    "DoorKey": "MiniGrid-DoorKey-6x6-v0",
    "MultiRoom": "MiniGrid-MultiRoom-N4-S5-v0",
}

STRATEGIES = [
    "baseline",
    "action_bonus",
    "state_bonus",
    "reward_shaping",
    "potential_shaping",
    "high_entropy",
]

TOTAL_TIMESTEPS = 512_000
EVAL_EPISODES = 100
SEEDS = [0, 1, 2]

RESULT_DIR = "results"
MODEL_DIR = "models"
LOG_DIR = "logs"


def resolve_trained_model_path(env_name: str, strategy: str, seed: int) -> str:
    """
    Prefer the EvalCallback checkpoint (best validation return during training)
    over the final weights. PPO can regress near the end of training.
    """

    base = os.path.join(MODEL_DIR, env_name, strategy, f"seed_{seed}")
    best = os.path.join(base, "best_model.zip")
    final = os.path.join(base, "final_model.zip")
    if os.path.isfile(best):
        return best
    return final
