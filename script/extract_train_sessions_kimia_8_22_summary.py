#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


EID_RE = re.compile(r"^EID:\s*(?P<eid>\S+)")
NUM_SAMPLES_FROM_EID_RE = re.compile(r"size_(?P<num_samples>\d+)")
TRAIN_ROWS_RE = re.compile(r"^\s*num_rows:\s*(?P<num_rows>\d+)")
TRAIN_DATASET_SIZE_RE = re.compile(r"^Train dataset size:\s*(?P<size>\d+)")
TRAIN_LOSS_RE = re.compile(
    r"^epoch:\s*(?P<epoch>\d+)\s+train loss:\s*(?P<train_loss>[-+eE0-9.]+)"
)
EVAL_RE = re.compile(
    r"^epoch:\s*(?P<epoch>\d+)\s+eval loss:\s*(?P<eval_loss>[-+eE0-9.]+)\s+r2:\s*(?P<r2>[-+eE0-9.]+)"
)
BEST_R2_RE = re.compile(
    r"^epoch:\s*(?P<epoch>\d+)\s+best eval trial avg r2:\s*(?P<best_r2>[-+eE0-9.]+)"
)


def parse_log(path: Path) -> dict:
    eid = None
    num_samples = None
    fallback_train_dataset_size = None
    epoch_metrics = {}
    in_train_dataset_block = False
    best_epoch = None
    best_r2 = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()

        match = EID_RE.match(line)
        if match:
            eid = match.group("eid")
            eid_num_samples = NUM_SAMPLES_FROM_EID_RE.search(eid)
            if eid_num_samples:
                num_samples = int(eid_num_samples.group("num_samples"))
            continue

        if line == "train: Dataset({":
            in_train_dataset_block = True
            continue

        if in_train_dataset_block:
            match = TRAIN_ROWS_RE.match(line)
            if match and num_samples is None:
                num_samples = int(match.group("num_rows"))
            if line == "})":
                in_train_dataset_block = False
            continue

        match = TRAIN_DATASET_SIZE_RE.match(line)
        if match:
            fallback_train_dataset_size = int(match.group("size"))
            continue

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
            continue

        match = BEST_R2_RE.match(line)
        if match:
            epoch = int(match.group("epoch"))
            candidate_r2 = float(match.group("best_r2"))
            if best_r2 is None or candidate_r2 > best_r2:
                best_epoch = epoch
                best_r2 = candidate_r2

    if eid is None:
        raise ValueError(f"Could not find EID in {path}")

    if num_samples is None:
        num_samples = fallback_train_dataset_size

    if num_samples is None:
        raise ValueError(f"Could not determine Num Samples for {path}")

    if best_epoch is None:
        raise ValueError(f"Could not determine best eval avg r2 epoch for {path}")

    best_metrics = epoch_metrics.get(best_epoch, {})
    if (
        "train_loss" not in best_metrics
        or "eval_loss" not in best_metrics
        or "r2" not in best_metrics
    ):
        raise ValueError(
            f"Missing train/eval metrics for best epoch {best_epoch} in {path}"
        )

    return {
        "source_file": path.name,
        "eid": eid,
        "num_samples": num_samples,
        "best_eval_avg_r2_epoch": best_epoch,
        "train_loss": best_metrics["train_loss"],
        "eval_loss": best_metrics["eval_loss"],
        "r2": best_metrics["r2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract EID, Num Samples, and best-eval-r2 epoch metrics from "
            "train-sessions-kimia-8-22 log files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("script"),
        help="Directory containing train-sessions-kimia-8-22-*.out files.",
    )
    parser.add_argument(
        "--pattern",
        default="train-sessions-kimia-8-22-*.out",
        help="Glob pattern used to find input files inside --input-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("script/train-sessions-kimia-8-22-summary.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    log_paths = sorted(args.input_dir.glob(args.pattern))
    if not log_paths:
        raise SystemExit(
            f"No files matched {args.pattern!r} in {args.input_dir.resolve()}"
        )

    rows = [parse_log(path) for path in log_paths]
    rows.sort(key=lambda row: (row["num_samples"], row["eid"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "eid",
                "num_samples",
                "best_eval_avg_r2_epoch",
                "train_loss",
                "eval_loss",
                "r2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
