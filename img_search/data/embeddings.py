"""Utilities for loading embedding datasets from Hugging Face."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


@dataclass(slots=True)
class EmbeddingDatasetSpec:
    """Specification for locating embedding vectors in a dataset."""

    dataset_name: str | None = None
    dataset_config: str | None = None
    data_files: Any | None = None
    split: str | None = "train"
    revision: str | None = None
    load_from_disk: str | None = None
    id_column: str = "id"
    embedding_column: str = "embedding"
    metadata_columns: list[str] = field(default_factory=list)
    read_batch_size: int | None = None
    memmap_path: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_name and not self.load_from_disk:
            raise ValueError("Either dataset_name or load_from_disk must be provided")
        if self.dataset_name and self.load_from_disk:
            raise ValueError("Specify only one of dataset_name or load_from_disk")
        if not self.id_column:
            raise ValueError("id_column must be provided")
        if not self.embedding_column:
            raise ValueError("embedding_column must be provided")


@dataclass(slots=True)
class QueryDatasetSpec(EmbeddingDatasetSpec):
    """Embedding spec with relevance labels for query evaluation."""

    relevance_column: str | None = None


def load_embedding_dataset(spec: EmbeddingDatasetSpec) -> Dataset:
    """Load a Hugging Face dataset or dataset split containing embeddings."""

    dataset: Dataset | DatasetDict
    if spec.load_from_disk:
        dataset = load_from_disk(spec.load_from_disk)
    else:
        kwargs: dict[str, Any] = {}
        if spec.data_files is not None:
            kwargs["data_files"] = spec.data_files
        if spec.revision is not None:
            kwargs["revision"] = spec.revision
        dataset = load_dataset(
            spec.dataset_name,  # type: ignore[arg-type]
            spec.dataset_config,
            split=spec.split,
            **kwargs,
        )
        if isinstance(dataset, Dataset):
            return dataset
    if isinstance(dataset, DatasetDict):
        if spec.split is None:
            raise ValueError("Split must be set when loading a DatasetDict from disk")
        if spec.split not in dataset:
            raise KeyError(f"Split '{spec.split}' not found in dataset")
        return dataset[spec.split]
    if spec.split is not None and isinstance(dataset, Dataset):
        return dataset
    return dataset  # type: ignore[return-value]


def extract_embeddings(
    dataset: Dataset,
    *,
    id_column: str,
    embedding_column: str,
    batch_size: int | None = None,
    memmap_path: str | Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return IDs and vectors from a dataset containing embeddings."""

    def _empty_embedding_matrix() -> np.ndarray:
        feature = getattr(dataset, "features", None)
        column_feature = None
        width: int | None = None
        if feature and embedding_column in feature:
            column_feature = feature[embedding_column]
            length = getattr(column_feature, "length", None)
            if isinstance(length, int):
                width = length
            else:
                shape = getattr(column_feature, "shape", None)
                if isinstance(shape, tuple | list) and shape:
                    last_axis = shape[-1]
                    if isinstance(last_axis, int):
                        width = last_axis
        if (
            width is None
            and column_feature is not None
            and hasattr(column_feature, "dtype")
        ):
            width = 1
        if width is None:
            width = 0
        return np.empty((0, width), dtype="float32")

    if id_column not in dataset.column_names:
        raise KeyError(f"Column '{id_column}' not found in dataset")
    if embedding_column not in dataset.column_names:
        raise KeyError(f"Column '{embedding_column}' not found in dataset")

    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        identifiers: list[str] = []
        batches: list[np.ndarray] = []
        memmap: np.memmap | None = None
        memmap_length: int | None = None
        if memmap_path is not None:
            try:
                memmap_length = len(dataset)
            except TypeError as exc:  # pragma: no cover - streaming datasets
                raise TypeError(
                    "memmap_path requires a dataset with a known length"
                ) from exc
            if memmap_length < 0:
                raise ValueError("Dataset length must be non-negative")

        def _ensure_memmap(total_rows: int, width: int) -> np.memmap:
            nonlocal memmap
            if memmap is not None:
                return memmap
            if memmap_path is None:
                tmp = tempfile.NamedTemporaryFile(suffix=".memmap", delete=False)
                path = Path(tmp.name)
                tmp.close()
            else:
                path = Path(memmap_path)
                path.parent.mkdir(parents=True, exist_ok=True)
            memmap = np.memmap(
                path,
                mode="w+",
                dtype="float32",
                shape=(total_rows, width),
            )
            return memmap

        offset = 0
        iterator = getattr(dataset, "iter", None)
        if callable(iterator):
            for batch in iterator(batch_size=batch_size):
                identifiers.extend(str(value) for value in batch[id_column])
                batch_vectors = np.asarray(batch[embedding_column], dtype="float32")
                if batch_vectors.ndim == 1:
                    batch_vectors = np.expand_dims(batch_vectors, axis=-1)
                if memmap_path is not None:
                    if memmap_length is None:
                        raise ValueError("Dataset length must be known for memmap")
                    if batch_vectors.ndim != 2:
                        raise ValueError("Embedding column must contain 2D vectors")
                    mm = _ensure_memmap(memmap_length, batch_vectors.shape[1])
                    stop = offset + batch_vectors.shape[0]
                    mm[offset:stop] = batch_vectors
                    offset = stop
                else:
                    batches.append(batch_vectors)
        else:
            total = len(dataset)
            for start in range(0, total, batch_size):
                stop = min(start + batch_size, total)
                slice_batch = dataset[start:stop]
                identifiers.extend(str(value) for value in slice_batch[id_column])
                batch_vectors = np.asarray(
                    slice_batch[embedding_column], dtype="float32"
                )
                if batch_vectors.ndim == 1:
                    batch_vectors = np.expand_dims(batch_vectors, axis=-1)
                if memmap_path is not None:
                    if batch_vectors.ndim != 2:
                        raise ValueError("Embedding column must contain 2D vectors")
                    mm = _ensure_memmap(total, batch_vectors.shape[1])
                    mm[start:stop] = batch_vectors
                else:
                    batches.append(batch_vectors)
        if memmap is not None:
            if memmap_length is not None and offset != 0 and offset != memmap_length:
                memmap = memmap[:offset]
            memmap.flush()
            return identifiers, memmap
        if not batches:
            return identifiers, _empty_embedding_matrix()
        vectors = np.concatenate(batches, axis=0)
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=-1)
        if vectors.ndim != 2:
            raise ValueError("Embedding column must contain 2D vectors")
        return identifiers, vectors

    identifiers = [str(value) for value in dataset[id_column]]
    if not identifiers:
        return identifiers, _empty_embedding_matrix()
    vectors = np.asarray(dataset[embedding_column], dtype="float32")
    if vectors.ndim == 1:
        vectors = np.expand_dims(vectors, axis=-1)
    if vectors.ndim != 2:
        raise ValueError("Embedding column must contain 2D vectors")
    return identifiers, vectors


def extract_relevance(
    dataset: Dataset,
    *,
    relevance_column: str,
) -> list[Sequence[str] | str | None]:
    """Pull relevance labels from the dataset when available."""

    if relevance_column not in dataset.column_names:
        raise KeyError(f"Column '{relevance_column}' not found in dataset")
    return dataset[relevance_column]
