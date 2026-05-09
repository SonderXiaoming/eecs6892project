#!/usr/bin/env bash
set -e

TIMESTEPS=${1:-300000}
EPISODES=${2:-50}

for env in FourRooms DoorKey MultiRoom
do
  for strategy in baseline action_bonus state_bonus reward_shaping high_entropy
  do
    for seed in 0 1 2
    do
      echo "Training env=$env strategy=$strategy seed=$seed timesteps=$TIMESTEPS"
      uv run train.py --env "$env" --strategy "$strategy" --seed "$seed" --timesteps "$TIMESTEPS"

      echo "Evaluating env=$env strategy=$strategy seed=$seed episodes=$EPISODES"
      uv run evaluate.py --env "$env" --strategy "$strategy" --seed "$seed" --episodes "$EPISODES"
    done
  done
done

uv run plot_results.py
