# evaluate.py

import os
import argparse
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import ENV_CONFIGS, STRATEGIES, EVAL_EPISODES, RESULT_DIR
from envs import make_env


def evaluate_model(env_name: str, strategy: str, seed: int, episodes: int):
    env_id = ENV_CONFIGS[env_name]

    env = DummyVecEnv([
        lambda: make_env(env_id, strategy=strategy, seed=seed + 20_000)
    ])

    model_file = f"models/{env_name}/{strategy}/seed_{seed}/final_model.zip"
    model = PPO.load(model_file)

    returns = []
    lengths = []
    successes = []

    for ep in range(episodes):
        obs = env.reset()
        done = False

        ep_return = 0.0
        ep_len = 0
        success = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)

            done = bool(dones[0])
            r = float(reward[0])
            ep_return += r
            ep_len += 1

            # MiniGrid success usually corresponds to positive terminal reward.
            if done and r > 0:
                success = 1

        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(success)

    env.close()

    result = {
        "env": env_name,
        "strategy": strategy,
        "seed": seed,
        "avg_return": np.mean(returns),
        "std_return": np.std(returns),
        "success_rate": np.mean(successes),
        "avg_episode_length": np.mean(lengths),
        "std_episode_length": np.std(lengths),
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="DoorKey", choices=list(ENV_CONFIGS.keys()))
    parser.add_argument("--strategy", type=str, default="baseline", choices=STRATEGIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)

    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    result = evaluate_model(
        env_name=args.env,
        strategy=args.strategy,
        seed=args.seed,
        episodes=args.episodes,
    )

    df = pd.DataFrame([result])
    out_file = f"{RESULT_DIR}/eval_{args.env}_{args.strategy}_seed{args.seed}.csv"
    df.to_csv(out_file, index=False)

    print(df)


if __name__ == "__main__":
    main()
