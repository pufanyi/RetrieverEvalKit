#!/usr/bin/env python3
"""Embed INQUIRE query texts with SigLIP and export Hugging Face parquet shards."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from img_search.embedding.siglip import SiglipEncoder

DEFAULT_ANNOTATIONS = (
    "https://raw.githubusercontent.com/inquire-benchmark/INQUIRE/"
    "refs/heads/main/data/inquire/inquire_annotations.csv"
)
DEFAULT_TEST_QUERIES = (
    "https://raw.githubusercontent.com/inquire-benchmark/INQUIRE/"
    "refs/heads/main/data/inquire/inquire_queries_test.csv"
)
DEFAULT_VAL_QUERIES = (
    "https://raw.githubusercontent.com/inquire-benchmark/INQUIRE/"
    "refs/heads/main/data/inquire/inquire_queries_val.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SigLIP text embeddings for INQUIRE queries."
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=DEFAULT_ANNOTATIONS,
        help="CSV with query_id to image_path relevance mappings.",
    )
    parser.add_argument(
        "--queries-test",
        type=str,
        default=DEFAULT_TEST_QUERIES,
        help="CSV containing test split query metadata.",
    )
    parser.add_argument(
        "--queries-val",
        type=str,
        default=DEFAULT_VAL_QUERIES,
        help="CSV containing validation split query metadata.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["test", "val"],
        choices=["test", "val"],
        help="Subset of splits to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inquire/query_embeddings"),
        help="Directory where parquet shards will be written.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="queries",
        help="Subdirectory to mirror Hugging Face config naming.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Maximum number of rows per parquet shard.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used for SigLIP encoding.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="SigLIP checkpoint identifier.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="inquire_siglip",
        help="Dataset identifier recorded in the parquet metadata.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g. 'cuda', 'cpu'); defaults to auto detection.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Transformers device_map passed to SigLIP when loading.",
    )
    parser.add_argument(
        "--normalise",
        action="store_true",
        help="L2 normalise embeddings before serialisation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shards when present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of queries processed per split (for smoke tests).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload generated shards to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Target Hugging Face dataset repo (e.g. user/inquire-siglip-so400m-queries).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token used for authentication.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Mark the pushed dataset as private.",
    )
    parser.add_argument(
        "--hf-commit-message",
        type=str,
        default=None,
        help="Optional commit message applied during upload.",
    )
    return parser.parse_args()


def _load_dataframe(path: str | Path) -> pd.DataFrame:
    logging.info("Loading CSV: %s", path)
    df = pd.read_csv(path)
    if df.columns[0] == "Unnamed: 0":
        df = df.drop(columns=df.columns[0])
    return df


def _load_annotations(path: str | Path) -> dict[int, list[str]]:
    annotations = _load_dataframe(path)
    required = {"query_id", "image_path"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotation columns missing: {', '.join(sorted(missing))}")
    grouped = (
        annotations.groupby("query_id")["image_path"]
        .apply(lambda series: sorted(set(str(item) for item in series)))
        .to_dict()
    )
    return {int(query_id): paths for query_id, paths in grouped.items()}


def _prepare_queries(path: str | Path) -> pd.DataFrame:
    queries = _load_dataframe(path)
    if "query_id" not in queries or "query_text" not in queries:
        raise ValueError("Query CSV must include 'query_id' and 'query_text'.")
    queries = queries.drop_duplicates(subset="query_id").reset_index(drop=True)
    queries["query_id"] = queries["query_id"].astype(int)
    queries["query_text"] = queries["query_text"].astype(str).str.strip()
    return queries


def _chunk_ranges(total: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def _l2_normalise(matrix: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-9
    norms = matrix.norm(dim=1, keepdim=True).clamp(min=epsilon)
    return matrix / norms


def _embed_queries(
    encoder: SiglipEncoder,
    texts: list[str],
    *,
    batch_size: int,
    normalise: bool,
) -> np.ndarray:
    encoder.build()
    vectors: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        with torch.no_grad():
            encoded = encoder.batch_encode(texts=batch)
        if isinstance(encoded, list):
            encoded = encoded[0]
        encoded = encoded.detach().to(dtype=torch.float32, device="cpu")
        if normalise:
            encoded = _l2_normalise(encoded)
        vectors.append(encoded)
    if not vectors:
        return np.zeros((0, 0), dtype="float16")
    matrix = torch.cat(vectors, dim=0)
    return matrix.to(dtype=torch.float16).numpy()


def _build_arrow_table(
    queries: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    model_name: str,
    dataset_name: str,
    split: str,
) -> pa.Table:
    if len(queries) != len(embeddings):
        raise ValueError("Row count mismatch between queries and embeddings.")
    embedding_dim = embeddings.shape[1]
    ids = pa.array(queries["query_id"].astype(str).tolist(), type=pa.string())
    query_ids = pa.array(queries["query_id"].tolist(), type=pa.int64())
    texts = pa.array(queries["query_text"].tolist(), type=pa.string())
    model = pa.array([model_name] * len(queries), type=pa.string())
    dataset = pa.array([dataset_name] * len(queries), type=pa.string())
    splits = pa.array([split] * len(queries), type=pa.string())

    embedding_flat = pa.array(
        np.ascontiguousarray(embeddings).reshape(-1),
        type=pa.float16(),
    )
    embedding_array = pa.FixedSizeListArray.from_arrays(embedding_flat, embedding_dim)

    image_lists = pa.array(queries["image_id"].tolist(), type=pa.list_(pa.string()))

    def _optional_column(name: str) -> pa.Array | None:
        if name in queries.columns:
            values = queries[name].fillna("").astype(str).tolist()
            return pa.array(values, type=pa.string())
        return None

    optional_cols = {
        "supercategory": _optional_column("supercategory"),
        "category": _optional_column("category"),
        "iconic_group": _optional_column("iconic_group"),
    }

    arrays = [
        ids,
        query_ids,
        texts,
        model,
        dataset,
        splits,
        image_lists,
        embedding_array,
    ]
    names = [
        "id",
        "query_id",
        "query_text",
        "model_name",
        "dataset_name",
        "split",
        "image_id",
        "embedding",
    ]
    for name, array in optional_cols.items():
        if array is not None:
            arrays.append(array)
            names.append(name)

    return pa.Table.from_arrays(arrays, names=names)


def _write_shards(
    table: pa.Table,
    *,
    output_dir: Path,
    split: str,
    chunk_size: int,
    overwrite: bool,
) -> list[Path]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = table.num_rows
    shard_paths: list[Path] = []
    if total_rows == 0:
        logging.warning("Split '%s' produced no rows; no shards written.", split)
        return shard_paths
    num_shards = (total_rows - 1) // chunk_size + 1
    shard_index = 0
    for start, stop in _chunk_ranges(total_rows, chunk_size):
        slice_table = table.slice(start, stop - start)
        filename = f"{split}-{shard_index:05d}-of-{num_shards:05d}.parquet"
        shard_path = output_dir / filename
        if shard_path.exists() and not overwrite:
            raise FileExistsError(
                f"Shard {shard_path} exists. Use --overwrite to replace."
            )
        pq.write_table(slice_table, shard_path)
        logging.info(
            "Wrote %d rows to %s",
            slice_table.num_rows,
            shard_path,
        )
        shard_paths.append(shard_path)
        shard_index += 1
    return shard_paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    annotations = _load_annotations(args.annotations)
    split_sources: dict[str, str] = {}
    if args.queries_test:
        split_sources["test"] = args.queries_test
    if args.queries_val:
        split_sources["val"] = args.queries_val

    device_map = args.device_map
    if isinstance(device_map, str) and device_map.lower() == "none":
        device_map = None

    encoder = SiglipEncoder(
        model_name=args.model_name,
        device_map=device_map,
        device=args.device,
    )

    all_shards: dict[str, list[Path]] = {}
    config_dir = args.output_dir / args.config_name

    for split in args.splits:
        source = split_sources.get(split)
        if source is None:
            logging.warning("No CSV provided for split '%s'; skipping.", split)
            continue
        queries = _prepare_queries(source)
        if args.limit is not None:
            if args.limit <= 0:
                logging.info(
                    "Split '%s' limit %d provides no rows; skipping.", split, args.limit
                )
                continue
            queries = queries.head(args.limit)
        queries["image_id"] = queries["query_id"].map(annotations).apply(
            lambda value: value if isinstance(value, list) else []
        )
        texts = queries["query_text"].tolist()
        logging.info("Encoding %d queries for split '%s'...", len(texts), split)
        embeddings = _embed_queries(
            encoder,
            texts,
            batch_size=args.batch_size,
            normalise=args.normalise,
        )
        table = _build_arrow_table(
            queries,
            embeddings,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            split=split,
        )
        shards = _write_shards(
            table,
            output_dir=config_dir,
            split=split,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
        all_shards[split] = shards

    if args.push_to_hub:
        if args.hf_repo_id is None:
            raise ValueError("--hf-repo-id is required when using --push-to-hub.")
        try:
            from scripts.prepare_data.convert_inquire_to_hf import push_to_hub
        except ImportError as exc:  # pragma: no cover - helper lives alongside script
            raise ImportError(
                "Unable to import push_to_hub helper from convert_inquire_to_hf.py"
            ) from exc
        for split, shards in all_shards.items():
            push_to_hub(
                shards=shards,
                split=split,
                repo_id=args.hf_repo_id,
                config_name=args.config_name,
                token=args.hf_token,
                private=args.hf_private,
                commit_message=args.hf_commit_message,
            )


if __name__ == "__main__":
    main()
