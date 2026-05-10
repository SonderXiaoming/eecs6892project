# plot_heatmap.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from config import ENV_CONFIGS, resolve_trained_model_path
from envs import make_env


def find_heatmap_wrapper(env):
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "position_counts"):
            return current
        current = current.env
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="DoorKey")
    parser.add_argument("--strategy", type=str, default="state_bonus")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)

    args = parser.parse_args()

    env_id = ENV_CONFIGS[args.env]

    env = make_env(
        env_id,
        strategy=args.strategy,
        seed=args.seed + 30_000,
        record_heatmap=True,
    )

    model_file = resolve_trained_model_path(args.env, args.strategy, args.seed)
    model = PPO.load(model_file)

    for _ in range(args.episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

    heatmap_wrapper = find_heatmap_wrapper(env)
    if heatmap_wrapper is None or len(heatmap_wrapper.position_counts) == 0:
        raise RuntimeError("No heatmap data collected.")

    counts = heatmap_wrapper.position_counts

    max_x = max(pos[0] for pos in counts.keys()) + 1
    max_y = max(pos[1] for pos in counts.keys()) + 1

    heatmap = np.zeros((max_y, max_x))

    for (x, y), c in counts.items():
        heatmap[y, x] = c

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, origin="upper")
    plt.colorbar(label="Visit Count")
    plt.title(f"Exploration Heatmap: {args.env} / {args.strategy}")
    plt.tight_layout()

    out_file = f"figures/heatmap_{args.env}_{args.strategy}_seed{args.seed}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()

    print(f"Saved heatmap to {out_file}")


if __name__ == "__main__":
    main()
