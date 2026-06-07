#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

cache_dir = Path("/tmp/ibl_mtm_plot_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["train_loss", "eval_loss", "r2"]
COLORS = {
    "train_loss": "#1f77b4",
    "eval_loss": "#ff7f0e",
    "r2": "#2ca02c",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean and standard deviation of train/eval loss and r2 "
            "versus the number of samples in each fold."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("script/train-sessions-folds-summary.csv"),
        help="Input CSV created from the train-sessions-folds runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("script/train-sessions-folds-summary-plot.png"),
        help="Path for the output figure.",
    )
    return parser


def summarize_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    grouped = dataframe.groupby("num_samples")[METRICS].agg(["mean", "std"])
    grouped.columns = [
        f"{metric}_{statistic}" for metric, statistic in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index().sort_values("num_samples")

    for metric in METRICS:
        grouped[f"{metric}_std"] = grouped[f"{metric}_std"].fillna(0.0)

    return grouped


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 12), sharex=True)

    for axis, metric in zip(axes, METRICS):
        mean_values = summary[f"{metric}_mean"]
        std_values = summary[f"{metric}_std"]
        color = COLORS[metric]

        axis.plot(
            summary["num_samples"],
            mean_values,
            marker="o",
            linewidth=2,
            color=color,
            label=f"{metric} mean",
        )
        axis.fill_between(
            summary["num_samples"],
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
            color=color,
            label=f"{metric} std",
        )
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Number of samples")
    fig.suptitle("Fold Metrics by Number of Samples")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    dataframe = pd.read_csv(args.input)

    required_columns = {"num_samples", *METRICS}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SystemExit(f"Missing required columns in {args.input}: {missing}")

    summary = summarize_metrics(dataframe)
    plot_summary(summary, args.output)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
