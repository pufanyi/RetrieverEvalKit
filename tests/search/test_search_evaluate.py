from pathlib import Path

import numpy as np
from datasets import Dataset

from img_search.data.embeddings import EmbeddingDatasetSpec, QueryDatasetSpec
from img_search.search import BenchmarkSettings, SearchEvalConfig, run_search_evaluation


def _build_dataset(base: Path, name: str, data: dict[str, list]) -> str:
    dataset = Dataset.from_dict(data)
    target = base / name
    dataset.save_to_disk(target)
    return str(target)


def test_run_search_evaluation_from_disk(tmp_path) -> None:
    gallery_vectors = np.eye(3, dtype="float32").tolist()
    queries = np.eye(3, dtype="float32").tolist()

    gallery_path = _build_dataset(
        tmp_path,
        "gallery",
        {"image_id": [f"img-{i}" for i in range(3)], "embedding": gallery_vectors},
    )
    query_path = _build_dataset(
        tmp_path,
        "queries",
        {
            "query_id": [f"q-{i}" for i in range(3)],
            "embedding": queries,
            "relevant_ids": [[f"img-{i}"] for i in range(3)],
        },
    )

    config = SearchEvalConfig(
        image_dataset=EmbeddingDatasetSpec(
            load_from_disk=gallery_path,
            split=None,
            id_column="image_id",
            embedding_column="embedding",
        ),
        query_dataset=QueryDatasetSpec(
            load_from_disk=query_path,
            split=None,
            id_column="query_id",
            embedding_column="embedding",
            relevance_column="relevant_ids",
        ),
        evaluation=BenchmarkSettings(
            methods=[{"method": "flat", "metric": "l2"}], top_k=1, recall_at=[1]
        ),
    )

    results = run_search_evaluation(config)

    assert results[0]["method"] == "flat"
    assert results[0]["num_queries"] == 3
    assert results[0]["top_k"] == 1
    assert results[0]["recall@1"] == 1.0
    assert results[0]["accuracy"] == 1.0
