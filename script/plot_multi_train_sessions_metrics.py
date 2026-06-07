#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path

cache_dir = Path("/tmp/ibl_mtm_plot_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


TRAIN_LOSS_RE = re.compile(
    r"^epoch:\s*(?P<epoch>\d+)\s+train loss:\s*(?P<train_loss>[-+eE0-9.]+)"
)
EVAL_RE = re.compile(
    r"^epoch:\s*(?P<epoch>\d+)\s+eval loss:\s*(?P<eval_loss>[-+eE0-9.]+)\s+r2:\s*(?P<r2>[-+eE0-9.]+)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot train loss, eval loss, and r2 across epochs from a "
            "multi-train-sessions log file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("script/multi-train-sessions-2172168.out"),
        help="Path to the multi-train-sessions output log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("script/multi-train-sessions-2172168-metrics.png"),
        help="Path for the output figure.",
    )
    return parser


def parse_log(path: Path) -> pd.DataFrame:
    epoch_metrics = {}

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()

        match = TRAIN_LOSS_RE.match(line)
        if match:
            epoch = int(match.group("epoch"))
            epoch_metrics.setdefault(epoch, {})["train_loss"] = float(
                match.group("train_loss")
            )
            continue

        match = EVAL_RE.match(line)
        if match:
            epoch = int(match.group("epoch"))
            metrics = epoch_metrics.setdefault(epoch, {})
            metrics["eval_loss"] = float(match.group("eval_loss"))
            metrics["r2"] = float(match.group("r2"))

    dataframe = (
        pd.DataFrame.from_dict(epoch_metrics, orient="index")
        .rename_axis("epoch")
        .reset_index()
        .sort_values("epoch")
    )

    required_columns = {"epoch", "train_loss", "eval_loss", "r2"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required metrics in {path}: {missing}")

    return dataframe


def plot_metrics(dataframe: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(
        dataframe["epoch"],
        dataframe["train_loss"],
        color="#1f77b4",
        linewidth=2,
        label="Train Loss",
    )
    axes[0].plot(
        dataframe["epoch"],
        dataframe["eval_loss"],
        color="#ff7f0e",
        linewidth=2,
        label="Eval Loss",
    )
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        dataframe["epoch"],
        dataframe["r2"],
        color="#2ca02c",
        linewidth=2,
    )
    axes[1].set_ylabel("R2")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Multi-Train Sessions Metrics: {output_path.stem}")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    dataframe = parse_log(args.input)
    plot_metrics(dataframe, args.output)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
