# plot_results.py

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULT_DIR


def load_results():
    files = glob.glob(f"{RESULT_DIR}/eval_*.csv")
    if not files:
        raise FileNotFoundError("No evaluation CSV files found.")

    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def plot_metric(df, metric, ylabel, out_file):
    grouped = df.groupby(["env", "strategy"])[metric].agg(["mean", "std"]).reset_index()

    envs = grouped["env"].unique()

    for env in envs:
        sub = grouped[grouped["env"] == env]

        plt.figure(figsize=(8, 5))
        plt.bar(
            sub["strategy"],
            sub["mean"],
            yerr=sub["std"],
            capsize=5,
        )
        plt.title(f"{metric} on {env}")
        plt.ylabel(ylabel)
        plt.xlabel("Exploration Strategy")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        os.makedirs("figures", exist_ok=True)
        path = f"figures/{env}_{out_file}"
        plt.savefig(path, dpi=200)
        plt.close()

        print(f"Saved: {path}")


def main():
    df = load_results()

    plot_metric(
        df,
        metric="success_rate",
        ylabel="Success Rate",
        out_file="success_rate.png",
    )

    plot_metric(
        df,
        metric="avg_return",
        ylabel="Average Return",
        out_file="avg_return.png",
    )

    plot_metric(
        df,
        metric="avg_episode_length",
        ylabel="Average Episode Length",
        out_file="episode_length.png",
    )


if __name__ == "__main__":
    main()
