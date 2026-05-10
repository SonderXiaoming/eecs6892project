# config.py

ENV_CONFIGS = {
    "FourRooms": "MiniGrid-FourRooms-v0",
    "DoorKey": "MiniGrid-DoorKey-5x5-v0",
    "MultiRoom": "MiniGrid-MultiRoom-N4-S5-v0",
}

STRATEGIES = [
    "baseline",
    "action_bonus",
    "state_bonus",
    "reward_shaping",
    "high_entropy",
]

TOTAL_TIMESTEPS = 300_000
EVAL_EPISODES = 50
SEEDS = [0, 1, 2]

RESULT_DIR = "results"
MODEL_DIR = "models"
LOG_DIR = "logs"
