from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from img_search.data.embeddings import extract_embeddings


def _build_dataset(num_rows: int = 4, dim: int = 3) -> Dataset:
    rows = [[float(row + offset) for offset in range(dim)] for row in range(num_rows)]
    data = {"id": list(range(num_rows)), "embedding": rows}
    return Dataset.from_dict(data)


def test_extract_embeddings_default_batch():
    dataset = _build_dataset(num_rows=3, dim=2)

    ids, vectors = extract_embeddings(
        dataset, id_column="id", embedding_column="embedding"
    )

    assert ids == ["0", "1", "2"]
    assert vectors.shape == (3, 2)
    np.testing.assert_allclose(
        vectors, np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype="float32")
    )


def test_extract_embeddings_streaming_batches():
    dataset = _build_dataset(num_rows=5, dim=3)

    ids, vectors = extract_embeddings(
        dataset,
        id_column="id",
        embedding_column="embedding",
        batch_size=2,
    )

    assert ids == ["0", "1", "2", "3", "4"]
    assert vectors.shape == (5, 3)
    for index, expected in enumerate(dataset["embedding"]):
        np.testing.assert_allclose(
            vectors[index], np.asarray(expected, dtype="float32")
        )


def test_extract_embeddings_rejects_invalid_batch_size():
    dataset = _build_dataset(num_rows=1, dim=2)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        extract_embeddings(
            dataset,
            id_column="id",
            embedding_column="embedding",
            batch_size=0,
        )
