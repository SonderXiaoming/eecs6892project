# train.py

import os
import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import ENV_CONFIGS, STRATEGIES, TOTAL_TIMESTEPS, MODEL_DIR, LOG_DIR
from envs import make_env


def resolve_device(device_arg: str) -> str:
    """
    Resolve device option.

    auto:
        use cuda if available, otherwise cpu.
    cuda:
        require CUDA. Raises an error if CUDA is unavailable.
    cpu:
        force CPU.
    """

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --device cpu."
            )
        return "cuda"

    if device_arg == "cpu":
        return "cpu"

    raise ValueError(f"Unknown device option: {device_arg}")


def train_one(
    env_name: str,
    strategy: str,
    seed: int,
    total_timesteps: int,
    device_arg: str,
    verbose: int,
    show_progress: bool,
):
    env_id = ENV_CONFIGS[env_name]
    device = resolve_device(device_arg)

    train_env = DummyVecEnv([
        lambda s=seed + i: make_env(env_id, strategy=strategy, seed=s)
        for i in range(8)
    ])

    eval_env = DummyVecEnv(
        [lambda: make_env(env_id, strategy=strategy, seed=seed + 10_000)]
    )

    model_path = f"{MODEL_DIR}/{env_name}/{strategy}/seed_{seed}"
    log_path = f"{LOG_DIR}/{env_name}/{strategy}/seed_{seed}"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    if strategy == "high_entropy":
        ent_coef = 0.05
    else:
        ent_coef = 0.01

    print(
        f"[train] env={env_name}, strategy={strategy}, seed={seed}, "
        f"timesteps={total_timesteps}, device={device}"
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        verbose=verbose,
        seed=seed,
        tensorboard_log=log_path,
        device=device,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=show_progress,
    )

    final_model_file = f"{model_path}/final_model.zip"
    model.save(final_model_file)

    train_env.close()
    eval_env.close()

    print(f"[done] saved model to {final_model_file}")

    return final_model_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="DoorKey", choices=list(ENV_CONFIGS.keys())
    )
    parser.add_argument("--strategy", type=str, default="baseline", choices=STRATEGIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device. Use cuda for GPU, cpu for CPU, or auto.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="SB3 logging verbosity. 0 is quiet, 1 prints training tables.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm/rich progress bar.",
    )

    args = parser.parse_args()

    train_one(
        env_name=args.env,
        strategy=args.strategy,
        seed=args.seed,
        total_timesteps=args.timesteps,
        device_arg=args.device,
        verbose=args.verbose,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
