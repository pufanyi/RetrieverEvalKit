"""Utilities for loading embedding datasets from Hugging Face."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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
) -> tuple[list[str], np.ndarray]:
    """Return IDs and vectors from a dataset containing embeddings."""

    if id_column not in dataset.column_names:
        raise KeyError(f"Column '{id_column}' not found in dataset")
    if embedding_column not in dataset.column_names:
        raise KeyError(f"Column '{embedding_column}' not found in dataset")

    identifiers = [str(value) for value in dataset[id_column]]
    vectors = np.asarray(dataset[embedding_column], dtype="float32")
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
