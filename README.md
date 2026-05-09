# MiniGrid Exploration Project

This project compares simple exploration strategies in sparse-reward MiniGrid environments.

## Strategies

- `baseline`: PPO with default entropy coefficient.
- `high_entropy`: PPO with larger entropy coefficient.
- `action_bonus`: MiniGrid built-in `ActionBonus`.
- `state_bonus`: custom count-based intrinsic reward.
- `reward_shaping`: small step penalty to encourage shorter solutions.

## Important Fix

This version uses:

```python
ImgObsWrapper + FlattenObservation + PPO("MlpPolicy")
```

instead of:

```python
ImgObsWrapper + PPO("CnnPolicy")
```

because MiniGrid compact observations are usually `7x7x3`, while Stable-Baselines3's default CNN has an `8x8` first convolution kernel. That causes:

```text
Kernel size can't be greater than actual input size
```

## Install

```bash
pip install -r requirements.txt
```

or with uv:

```bash
uv pip install -r requirements.txt
```

## Quick Windows Test

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -Timesteps 10000 -Envs FourRooms -Strategies baseline -Seeds 0 -SkipHeatmap
```

## Full Run

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -SkipHeatmap
```

Heatmaps can be slow, so it is recommended to run them only for selected models.


## Quiet Logging and GPU

This version sets SB3 `verbose=0` by default to avoid printing PPO tables every few iterations.

Run with GPU:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -Device cuda -SkipHeatmap
```

Small GPU test:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -Device cuda -Timesteps 10000 -Envs FourRooms -Strategies baseline -Seeds 0 -SkipHeatmap
```

Fully quiet, no progress bar:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -Device cuda -NoProgress -Timesteps 10000 -Envs FourRooms -Strategies baseline -Seeds 0 -SkipHeatmap
```

To bring back SB3 training logs:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -UseUv -Device cuda -Verbose 1 -SkipHeatmap
```

Check CUDA:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
