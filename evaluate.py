# evaluate.py
# Evaluation script using RAW environment rewards.
#
# During training, action_bonus/state_bonus add intrinsic rewards.
# If we evaluate with the same bonus wrappers, avg_return and success_rate
# become inflated and do not represent true task completion.
#
# This version loads the trained model but evaluates it in the raw MiniGrid
# environment with strategy="baseline". Observation/action spaces are unchanged,
# so the policy remains compatible.

import os
import argparse
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import ENV_CONFIGS, STRATEGIES, EVAL_EPISODES, RESULT_DIR, resolve_trained_model_path
from envs import make_env


def evaluate_model(env_name: str, trained_strategy: str, seed: int, episodes: int):
    env_id = ENV_CONFIGS[env_name]

    # IMPORTANT:
    # Always evaluate on the raw environment, not on reward-bonus wrappers.
    # This makes success_rate and avg_return comparable across strategies.
    env = DummyVecEnv([
        lambda: make_env(env_id, strategy="baseline", seed=seed + 20_000)
    ])

    model_file = resolve_trained_model_path(env_name, trained_strategy, seed)
    model = PPO.load(model_file)

    returns = []
    lengths = []
    successes = []
    timeouts = []
    success_lengths = []
    failure_lengths = []

    for ep in range(episodes):
        obs = env.reset()
        done = False

        ep_return = 0.0
        ep_len = 0
        success = 0
        timeout = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)

            done = bool(dones[0])
            raw_reward = float(reward[0])
            info = infos[0]

            ep_return += raw_reward
            ep_len += 1

            # In VecEnv, terminal_observation and TimeLimit.truncated may appear in info.
            if done:
                time_limit_truncated = bool(info.get("TimeLimit.truncated", False))

                if raw_reward > 0 and not time_limit_truncated:
                    success = 1
                elif time_limit_truncated:
                    timeout = 1


        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(success)
        timeouts.append(timeout)
        
        if success:
            success_lengths.append(ep_len)
        else:
            failure_lengths.append(ep_len)
    env.close()

    return {
        "env": env_name,
        "strategy": trained_strategy,
        "seed": seed,
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "timeout_rate": float(np.mean(timeouts)),
        "avg_episode_length": float(np.mean(lengths)),
        "avg_success_length": float(np.mean(success_lengths)) if len(success_lengths) > 0 else float(np.nan),
        "avg_failure_length": float(np.mean(failure_lengths)) if len(failure_lengths) > 0 else float(np.nan),
        "std_episode_length": float(np.std(lengths)),
        "eval_reward_type": "raw_env_reward",
    }


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
        trained_strategy=args.strategy,
        seed=args.seed,
        episodes=args.episodes,
    )

    df = pd.DataFrame([result])
    out_file = f"{RESULT_DIR}/eval_{args.env}_{args.strategy}_seed{args.seed}.csv"
    df.to_csv(out_file, index=False)

    print(df)


if __name__ == "__main__":
    main()
