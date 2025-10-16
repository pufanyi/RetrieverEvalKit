import math

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from img_search.search import (
    FaissIndexConfig,
    FaissSearchIndex,
    benchmark_methods,
)


def _gpu_available() -> bool:
    return hasattr(faiss, "StandardGpuResources") and getattr(faiss, "get_num_gpus", lambda: 0)() > 0


def test_flat_search_returns_expected_order():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype="float32",
    )
    ids = ["east", "north", "diag"]
    index = FaissSearchIndex(dim=2, config={"method": "flat", "metric": "l2"})
    index.add_embeddings(ids, embeddings)

    results, elapsed = index.search([0.1, 0.9], top_k=2, return_time=True)

    assert [hit["id"] for hit in results] == ["north", "diag"]
    assert index.ntotal == len(ids)
    assert elapsed >= 0.0


def test_ivf_flat_trains_and_searches():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(32, 4)).astype("float32")
    ids = [f"vec-{i}" for i in range(len(embeddings))]
    index = FaissSearchIndex(
        dim=4,
        config={"method": "ivf_flat", "metric": "l2", "nlist": 4, "nprobe": 4},
    )
    index.add_embeddings(ids, embeddings)

    query = embeddings[0]
    results = index.search(query, top_k=3)

    assert results[0]["id"] == ids[0]
    assert index.ntotal == len(ids)


def test_benchmark_methods_reports_accuracy_and_timings():
    embeddings = np.eye(4, dtype="float32")
    queries = embeddings.copy()
    ids = [f"id-{i}" for i in range(len(embeddings))]
    configs = [
        {"method": "flat", "metric": "l2"},
        {"method": "ivf_flat", "metric": "l2", "nlist": 2, "nprobe": 2},
    ]
    ground_truth = [[identifier] for identifier in ids]

    summary = benchmark_methods(
        embeddings,
        queries,
        ids=ids,
        method_configs=configs,
        top_k=1,
        ground_truth=ground_truth,
        recall_points=[1],
    )

    assert len(summary) == len(configs)
    for row in summary:
        assert math.isclose(row["accuracy"], 1.0, rel_tol=1e-6)
        assert row["ntotal"] == len(ids)
        assert row["index_time"] >= 0
        assert row["search_time"] >= 0
        assert math.isclose(row["recall@1"], 1.0, rel_tol=1e-6)


@pytest.mark.skipif(not _gpu_available(), reason="FAISS GPU bindings unavailable")
def test_gpu_and_cpu_indices_return_equivalent_results():
    embeddings = np.array(
        [
            [0.9, 0.1],
            [0.0, 1.0],
            [0.7, 0.3],
        ],
        dtype="float32",
    )
    ids = ["a", "b", "c"]

    cpu_index = FaissSearchIndex(dim=2, config={"method": "flat", "metric": "cosine"})
    cpu_index.add_embeddings(ids, embeddings)
    cpu_results = cpu_index.search([0.8, 0.2], top_k=3)

    gpu_index = FaissSearchIndex(
        dim=2,
        config=FaissIndexConfig(method="flat", metric="cosine", use_gpu=True),
    )
    gpu_index.add_embeddings(ids, embeddings)
    gpu_results = gpu_index.search([0.8, 0.2], top_k=3)

    assert [hit["id"] for hit in gpu_results] == [hit["id"] for hit in cpu_results]
