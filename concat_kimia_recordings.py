#!/usr/bin/env python
"""Concatenate processed_kimia_8_22 recording parquet datasets.

Examples:
  python concat_kimia_recordings.py --concat-files recording1_all recording2_all recording3_all recording7_all --output-dir /tmp/rec1237/data
  python concat_kimia_recordings.py --variant all --sizes 2 3
  python concat_kimia_recordings.py --name rec123_all --recordings recording1_all recording2_all recording3_all
  python concat_kimia_recordings.py --combo rec12_all:recording1_all,recording2_all --combo rec34_all:recording3_all,recording4_all
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

DEFAULT_INPUT_ROOT = Path("/work/hdd/beml/ac136/processed_kimia_8_22")
DEFAULT_OUTPUT_ROOT = Path("/work/hdd/beml/ac136/processed_kimia_8_22_concat")
SPLITS = ("train", "val")

# Optional simple mode. Fill these in, or use the matching environment variables
# / CLI flags, to concatenate matching train.parquet and val.parquet files into
# OUTPUT_DIR/train.parquet and OUTPUT_DIR/val.parquet.
CONCAT_FILES: tuple[str, ...] = ()
OUTPUT_DIR: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate processed_kimia_8_22 recordings into single-session-style datasets."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--concat-files",
        nargs="+",
        default=None,
        help=(
            "Simple split-wise concat inputs. Each item may be a recording name under "
            "--input-root, a recording directory, a data directory, or a train/val parquet file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for --concat-files mode. Writes train.parquet and val.parquet directly here.",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="One explicit recording combination, for example recording1_all recording2_all.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output dataset name for --recordings. Defaults to the joined recording names.",
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        metavar="NAME:REC1,REC2",
        help="Named combination. Can be repeated.",
    )
    parser.add_argument(
        "--variant",
        choices=("all", "active"),
        default=None,
        help="Generate combinations from all recordings with this suffix.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Combination sizes to generate with --variant, for example --sizes 2 3.",
    )
    parser.add_argument(
        "--allow-mixed-shapes",
        action="store_true",
        help="Allow recordings with different spikes_sparse_shape values.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parquet files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    return parser.parse_args()


def parse_env_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.replace(os.pathsep, ",").split(",") if part.strip())


def safe_dataset_name(recordings: tuple[str, ...]) -> str:
    return "__".join(recordings)


def parse_combo(combo: str) -> tuple[str, tuple[str, ...]]:
    if ":" not in combo:
        raise ValueError(f"Combo must look like NAME:REC1,REC2, got {combo!r}")
    name, recordings_csv = combo.split(":", 1)
    recordings = tuple(part.strip() for part in recordings_csv.split(",") if part.strip())
    if not name or not recordings:
        raise ValueError(f"Combo must include both name and recordings, got {combo!r}")
    return name, recordings


def discover_recordings(input_root: Path, variant: str) -> list[str]:
    suffix = f"_{variant}"
    return sorted(path.name for path in input_root.iterdir() if path.is_dir() and path.name.endswith(suffix))


def build_jobs(args: argparse.Namespace) -> list[tuple[str, tuple[str, ...]]]:
    jobs: list[tuple[str, tuple[str, ...]]] = []

    if args.recordings:
        recordings = tuple(args.recordings)
        jobs.append((args.name or safe_dataset_name(recordings), recordings))

    for combo in args.combo:
        jobs.append(parse_combo(combo))

    if args.variant:
        if not args.sizes:
            raise ValueError("--variant requires --sizes unless --recordings or --combo supplies all jobs")
        recordings = discover_recordings(args.input_root, args.variant)
        for size in args.sizes:
            if size < 1:
                raise ValueError(f"Combination size must be >= 1, got {size}")
            for combo in itertools.combinations(recordings, size):
                jobs.append((safe_dataset_name(combo), combo))

    if not jobs:
        raise ValueError("No jobs requested. Use --recordings, --combo, or --variant with --sizes.")

    seen = set()
    unique_jobs = []
    for name, recordings in jobs:
        if name in seen:
            raise ValueError(f"Duplicate output dataset name: {name}")
        seen.add(name)
        unique_jobs.append((name, recordings))
    return unique_jobs


def split_path(input_root: Path, recording: str, split: str) -> Path:
    return input_root / recording / "data" / f"{split}.parquet"


def resolve_named_or_dir_split(input_root: Path, source: str, split: str) -> Path:
    source_path = Path(source).expanduser()
    candidates: list[Path] = []

    if source_path.is_absolute():
        candidates.extend(
            [
                source_path / "data" / f"{split}.parquet",
                source_path / f"{split}.parquet",
            ]
        )
    else:
        candidates.extend(
            [
                input_root / source_path / "data" / f"{split}.parquet",
                input_root / source_path / f"{split}.parquet",
                source_path / "data" / f"{split}.parquet",
                source_path / f"{split}.parquet",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {split}.parquet for {source!r}. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def source_label(source: str, path: Path) -> str:
    if path.name in {f"{split}.parquet" for split in SPLITS}:
        if path.parent.name == "data":
            return path.parent.parent.name
        return path.parent.name
    return Path(source).name


def resolve_concat_sources(input_root: Path, concat_files: tuple[str, ...]) -> dict[str, list[tuple[str, Path]]]:
    split_sources: dict[str, list[tuple[str, Path]]] = {split: [] for split in SPLITS}

    for source in concat_files:
        source_path = Path(source).expanduser()
        if source_path.suffix == ".parquet":
            if source_path.name not in {f"{split}.parquet" for split in SPLITS}:
                raise ValueError(f"Parquet input must be named train.parquet or val.parquet: {source}")
            if not source_path.is_absolute() and not source_path.exists():
                input_root_source_path = input_root / source_path
                if input_root_source_path.exists():
                    source_path = input_root_source_path
            if not source_path.exists():
                raise FileNotFoundError(f"Missing parquet input: {source_path}")
            split = source_path.stem
            split_sources[split].append((source_label(source, source_path), source_path))
            continue

        for split in SPLITS:
            path = resolve_named_or_dir_split(input_root, source, split)
            split_sources[split].append((source_label(source, path), path))

    missing_splits = [split for split, paths in split_sources.items() if not paths]
    if missing_splits:
        raise ValueError(f"No inputs found for split(s): {', '.join(missing_splits)}")

    return split_sources


def load_recording_split(input_root: Path, recording: str, split: str):
    from datasets import load_dataset

    path = split_path(input_root, recording, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing {split} split for {recording}: {path}")
    return load_dataset("parquet", data_files={split: str(path)})[split]


def load_parquet_split(path: Path, split: str):
    from datasets import load_dataset

    return load_dataset("parquet", data_files={split: str(path)})[split]


def first_sparse_shape(dataset) -> tuple[int, ...]:
    if len(dataset) == 0:
        raise ValueError("Cannot validate an empty dataset split")
    return tuple(dataset[0]["spikes_sparse_shape"])


def concatenate_job(
    input_root: Path,
    output_root: Path,
    name: str,
    recordings: tuple[str, ...],
    allow_mixed_shapes: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    output_data_dir = output_root / name / "data"
    output_files = {split: output_data_dir / f"{split}.parquet" for split in SPLITS}

    existing_outputs = [path for path in output_files.values() if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            f"Output exists for {name}: {existing_outputs}. Use --overwrite to replace it."
        )

    print(f"\n{name}")
    print(f"  recordings: {', '.join(recordings)}")

    if dry_run:
        for split, path in output_files.items():
            print(f"  would write {split}: {path}")
        return

    output_data_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "name": name,
        "recordings": list(recordings),
        "splits": {},
    }

    for split in SPLITS:
        from datasets import concatenate_datasets

        datasets = []
        shapes = {}
        split_lengths = {}
        for source_idx, recording in enumerate(recordings):
            dataset = load_recording_split(input_root, recording, split)
            shape = first_sparse_shape(dataset)
            shapes[recording] = list(shape)
            split_lengths[recording] = len(dataset)
            dataset = dataset.add_column("source_recording", [recording] * len(dataset))
            dataset = dataset.add_column("source_recording_idx", [source_idx] * len(dataset))
            datasets.append(dataset)

        unique_shapes = {tuple(shape) for shape in shapes.values()}
        if len(unique_shapes) > 1 and not allow_mixed_shapes:
            raise ValueError(
                f"{name} has mixed {split} spikes_sparse_shape values: {shapes}. "
                "Use --allow-mixed-shapes only if your downstream training can handle this."
            )

        concatenated = concatenate_datasets(datasets)
        concatenated.to_parquet(str(output_files[split]))
        summary["splits"][split] = {
            "num_rows": len(concatenated),
            "source_rows": split_lengths,
            "spikes_sparse_shape": shapes,
        }
        print(f"  wrote {split}: {len(concatenated)} rows -> {output_files[split]}")

    metadata_path = output_root / name / "concat_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote metadata: {metadata_path}")


def concatenate_files(
    input_root: Path,
    output_dir: Path,
    concat_files: tuple[str, ...],
    allow_mixed_shapes: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    split_sources = resolve_concat_sources(input_root, concat_files)
    output_files = {split: output_dir / f"{split}.parquet" for split in SPLITS}

    existing_outputs = [path for path in output_files.values() if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(f"Output exists: {existing_outputs}. Use --overwrite to replace it.")

    print(f"Input root: {input_root}")
    print(f"Output dir: {output_dir}")
    print(f"Inputs: {len(concat_files)}")

    for split, sources in split_sources.items():
        print(f"  {split}:")
        for label, path in sources:
            print(f"    {label}: {path}")
        if dry_run:
            print(f"    would write: {output_files[split]}")

    if dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "concat_files": list(concat_files),
        "splits": {},
    }

    for split, sources in split_sources.items():
        from datasets import concatenate_datasets

        datasets = []
        shapes = {}
        split_lengths = {}

        for source_idx, (label, path) in enumerate(sources):
            dataset = load_parquet_split(path, split)
            shape = first_sparse_shape(dataset)
            shapes[label] = list(shape)
            split_lengths[label] = len(dataset)
            dataset = dataset.add_column("source_recording", [label] * len(dataset))
            dataset = dataset.add_column("source_recording_idx", [source_idx] * len(dataset))
            datasets.append(dataset)

        unique_shapes = {tuple(shape) for shape in shapes.values()}
        if len(unique_shapes) > 1 and not allow_mixed_shapes:
            raise ValueError(
                f"Mixed {split} spikes_sparse_shape values: {shapes}. "
                "Use --allow-mixed-shapes only if your downstream training can handle this."
            )

        concatenated = concatenate_datasets(datasets)
        concatenated.to_parquet(str(output_files[split]))
        summary["splits"][split] = {
            "num_rows": len(concatenated),
            "source_rows": split_lengths,
            "spikes_sparse_shape": shapes,
            "source_files": {label: str(path) for label, path in sources},
        }
        print(f"  wrote {split}: {len(concatenated)} rows -> {output_files[split]}")

    metadata_path = output_dir / "concat_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote metadata: {metadata_path}")


def main() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")

    args = parse_args()
    concat_files = (
        tuple(args.concat_files)
        if args.concat_files
        else tuple(CONCAT_FILES) or parse_env_list(os.environ.get("CONCAT_FILES"))
    )
    output_dir = (
        args.output_dir
        or (Path(OUTPUT_DIR).expanduser() if OUTPUT_DIR else None)
        or (Path(os.environ["OUTPUT_DIR"]).expanduser() if os.environ.get("OUTPUT_DIR") else None)
    )

    if concat_files:
        legacy_requested = bool(args.recordings or args.combo or args.variant)
        if legacy_requested:
            raise ValueError("Use either --concat-files/CONCAT_FILES mode or the legacy combination options, not both.")
        if output_dir is None:
            raise ValueError("--concat-files/CONCAT_FILES mode requires --output-dir or OUTPUT_DIR.")
        concatenate_files(
            input_root=args.input_root,
            output_dir=output_dir,
            concat_files=concat_files,
            allow_mixed_shapes=args.allow_mixed_shapes,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        return

    jobs = build_jobs(args)
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Jobs: {len(jobs)}")

    for name, recordings in jobs:
        concatenate_job(
            input_root=args.input_root,
            output_root=args.output_root,
            name=name,
            recordings=recordings,
            allow_mixed_shapes=args.allow_mixed_shapes,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
