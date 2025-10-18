#!/usr/bin/env python3
"""Convert INQUIRE metadata and embeddings to a Hugging Face compatible format."""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_METADATA_DIR = Path("data/inquire/data/metadata")
DEFAULT_EMBEDDING_DIR = Path("data/inquire/data/img_emb")
DEFAULT_OUTPUT_DIR = Path("outputs/inquire/hf")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert INQUIRE image embeddings stored as NumPy/Parquet pairs into "
            "Parquet shards that can be loaded with "
            "`datasets.load_dataset('parquet', ...)`."
        )
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Directory containing metadata_*.parquet files.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
        help="Directory containing img_emb_*.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for Parquet shards in Hugging Face format.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name to embed in output filenames.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="jinaai/jina-embeddings-v4",
        help="Model identifier to store alongside embeddings.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="inquire",
        help="Dataset identifier to store alongside embeddings.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows to materialize per Parquet write chunk.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the total number of rows to convert (across all shards).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing Parquet shards in the output directory.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the generated shards to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Target Hugging Face dataset repository (e.g. user/inquire-embeddings).",
    )
    parser.add_argument(
        "--hf-config-name",
        type=str,
        default="images",
        help="Configuration name to publish when pushing to the Hub.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face API token; falls back to cached token if omitted.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create or update the Hub dataset as private.",
    )
    parser.add_argument(
        "--hf-commit-message",
        type=str,
        default=None,
        help="Optional commit message to use when pushing to the Hub.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the conversion pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    pairs = collect_shard_pairs(args.metadata_dir, args.embeddings_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No shard pairs found in {args.metadata_dir} and {args.embeddings_dir}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    remaining = args.limit
    total_written = 0
    shards_converted = 0

    for index, metadata_path, embedding_path in pairs:
        if remaining is not None and remaining <= 0:
            break

        output_path = shard_output_path(args.output_dir, args.split, index)
        written = convert_shard(
            metadata_path=metadata_path,
            embedding_path=embedding_path,
            output_path=output_path,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
            limit=remaining,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
        )
        total_written += written
        if written:
            shards_converted += 1
        if remaining is not None:
            remaining -= written

    logging.info(
        "Finished conversion. Wrote %d rows across %d shard(s).",
        total_written,
        shards_converted,
    )
    if args.push_to_hub:
        if args.hf_repo_id is None:
            raise ValueError("--hf-repo-id must be specified when using --push-to-hub.")
        shard_paths = sorted(args.output_dir.glob(f"{args.split}-*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No Parquet shards matching '{args.split}-*.parquet' found in "
                f"{args.output_dir}."
            )
        push_to_hub(
            shards=shard_paths,
            split=args.split,
            repo_id=args.hf_repo_id,
            config_name=args.hf_config_name,
            token=args.hf_token,
            private=args.hf_private,
            commit_message=args.hf_commit_message,
        )


def collect_shard_pairs(
    metadata_dir: Path, embeddings_dir: Path
) -> list[tuple[str, Path, Path]]:
    """Match metadata_* parquet shards with img_emb_* numpy shards."""

    def _index(path: Path, prefix: str) -> str:
        pattern = rf"{re.escape(prefix)}(\d+)$"
        match = re.search(pattern, path.stem)
        if not match:
            raise ValueError(f"Could not extract numeric suffix from {path.name}.")
        return match.group(1).lstrip("0") or "0"

    metadata_files = {
        _index(path, "metadata_"): path
        for path in sorted(metadata_dir.glob("metadata_*.parquet"))
    }
    embedding_files = {
        _index(path, "img_emb_"): path
        for path in sorted(embeddings_dir.glob("img_emb_*.npy"))
    }

    missing_in_embeddings = sorted(set(metadata_files) - set(embedding_files))
    missing_in_metadata = sorted(set(embedding_files) - set(metadata_files))
    if missing_in_embeddings:
        logging.warning(
            "Metadata shards without embeddings will be skipped: %s",
            ", ".join(missing_in_embeddings),
        )
    if missing_in_metadata:
        logging.warning(
            "Embedding shards without metadata will be skipped: %s",
            ", ".join(missing_in_metadata),
        )

    common_indices = sorted(
        set(metadata_files) & set(embedding_files), key=lambda value: int(value)
    )
    if not common_indices:
        raise FileNotFoundError("No matching metadata/img_emb shard pairs were found.")

    return [
        (index, metadata_files[index], embedding_files[index])
        for index in common_indices
    ]


def shard_output_path(output_dir: Path, split: str, index: str) -> Path:
    """Produce a deterministic output filename."""
    return output_dir / f"{split}-{int(index):02d}.parquet"


def convert_shard(
    *,
    metadata_path: Path,
    embedding_path: Path,
    output_path: Path,
    chunk_size: int,
    overwrite: bool,
    limit: int | None,
    model_name: str,
    dataset_name: str,
) -> int:
    """Convert a single Parquet/NumPy shard pair."""
    if output_path.exists():
        if overwrite:
            logging.debug("Overwriting existing shard %s.", output_path)
            output_path.unlink()
        else:
            logging.info("Skipping existing shard %s.", output_path)
            return 0

    logging.info("Converting shard %s -> %s.", metadata_path.name, output_path.name)
    metadata = pd.read_parquet(metadata_path, columns=["image_path"])
    embeddings = np.load(embedding_path, mmap_mode="r")

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Shard size mismatch: {metadata_path.name} has {len(metadata)} rows, "
            f"but {embedding_path.name} has {embeddings.shape[0]} embeddings."
        )

    total_rows = len(metadata)
    embedding_dim = embeddings.shape[1]
    rows_to_write = min(limit, total_rows) if limit is not None else total_rows

    if rows_to_write <= 0:
        return 0

    metadata_series = metadata["image_path"]
    writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        for start, stop in _chunk_ranges(rows_to_write, chunk_size):
            chunk_table = build_arrow_table(
                metadata_series.iloc[start:stop],
                embeddings[start:stop],
                embedding_dim=embedding_dim,
                model_name=model_name,
                dataset_name=dataset_name,
            )
            if chunk_table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path, chunk_table.schema, compression="zstd"
                )
            writer.write_table(chunk_table)
            rows_written += chunk_table.num_rows
            logging.debug("Wrote rows %d-%d for %s.", start, stop - 1, output_path.name)
            if limit is not None and rows_written >= rows_to_write:
                break
    finally:
        if writer is not None:
            writer.close()

    logging.info(
        "Finished shard %s with %d rows (embedding_dim=%d).",
        output_path.name,
        rows_written,
        embedding_dim,
    )
    return rows_written


def _chunk_ranges(total_rows: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    """Yield (start, stop) ranges covering total_rows."""
    for start in range(0, total_rows, chunk_size):
        stop = min(start + chunk_size, total_rows)
        yield start, stop


def build_arrow_table(
    paths: pd.Series,
    embeddings: np.ndarray,
    *,
    embedding_dim: int,
    model_name: str,
    dataset_name: str,
) -> pa.Table:
    """Build a PyArrow table for a single chunk."""
    if embeddings.shape[0] != len(paths):
        raise ValueError("Chunk row counts do not align.")
    if embeddings.shape[0] == 0:
        return pa.table({})

    num_rows = len(paths)
    ids = pa.array(paths.astype(str).tolist(), type=pa.string())
    embedding_flat = pa.array(
        np.ascontiguousarray(embeddings).reshape(-1),
        type=pa.float16(),
    )
    embedding_array = pa.FixedSizeListArray.from_arrays(embedding_flat, embedding_dim)
    model_array = pa.array([model_name] * num_rows, type=pa.string())
    dataset_array = pa.array([dataset_name] * num_rows, type=pa.string())

    return pa.Table.from_arrays(
        [ids, ids, model_array, dataset_array, embedding_array],
        names=["id", "image_path", "model_name", "dataset_name", "embedding"],
    )


def push_to_hub(
    *,
    shards: Sequence[Path],
    split: str,
    repo_id: str,
    config_name: str,
    token: str | None,
    private: bool,
    commit_message: str | None,
) -> None:
    """Upload generated Parquet shards to the Hugging Face Hub."""
    if not shards:
        raise ValueError("No Parquet shards were generated; nothing to upload.")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "datasets is required to push to the Hugging Face Hub."
        ) from exc

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "huggingface_hub is required to push to the Hugging Face Hub."
        ) from exc

    shard_paths = [str(path) for path in shards]
    logging.info(
        "Uploading %d shard(s) to %s (config=%s, split=%s).",
        len(shard_paths),
        repo_id,
        config_name,
        split,
    )
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )

    data_files = {split: shard_paths}
    logging.info("Loading dataset from local Parquet shards for validation...")
    dataset_dict = load_dataset("parquet", data_files=data_files)
    dataset_split = dataset_dict[split]
    logging.info("Pushing %d rows to the Hugging Face Hub...", len(dataset_split))
    dataset_split.push_to_hub(
        repo_id=repo_id,
        token=token,
        config_name=config_name,
        private=private,
        commit_message=commit_message,
        split=split,
    )
    logging.info("✅ Successfully pushed dataset to %s.", repo_id)


if __name__ == "__main__":
    main()
