import numpy as np

from img_search.search.faiss_search import (
    benchmark_bruteforce,
    benchmark_methods,
)
from img_search.search.hnswlib_search import HnswlibIndexConfig, HnswlibSearchIndex
from img_search.search.scann_search import ScannIndexConfig, ScannSearchIndex


def test_scann_search_index_returns_expected_neighbors() -> None:
    embeddings = np.eye(4, dtype="float32")
    ids = [f"img-{i}" for i in range(4)]

    index = ScannSearchIndex(
        embeddings.shape[1],
        config=ScannIndexConfig(num_neighbors=4),
    )
    index.add_embeddings(ids, embeddings)

    results = index.search(embeddings, top_k=1)

    assert len(results) == len(ids)
    assert all(row for row in results)
    assert results[0][0]["id"] == ids[0]


def test_hnswlib_search_index_returns_expected_neighbors() -> None:
    embeddings = np.eye(3, dtype="float32")
    ids = [f"img-{i}" for i in range(3)]

    index = HnswlibSearchIndex(
        embeddings.shape[1],
        config=HnswlibIndexConfig(metric="l2", m=4, ef_construction=50, ef_search=10),
    )
    index.add_embeddings(ids, embeddings)

    results = index.search(embeddings, top_k=1)

    assert [row[0]["id"] for row in results] == ids


def test_benchmark_methods_supports_multiple_backends() -> None:
    embeddings = np.eye(2, dtype="float32")
    ids = [f"img-{i}" for i in range(2)]
    configs = [
        {"backend": "faiss", "method": "flat", "metric": "l2"},
        {
            "backend": "scann",
            "method": "scann",
            "metric": "l2",
            "num_neighbors": 2,
        },
        {
            "backend": "hnswlib",
            "method": "hnsw",
            "metric": "l2",
            "m": 4,
            "ef_construction": 40,
            "ef_search": 10,
        },
    ]

    rows = benchmark_methods(
        embeddings,
        embeddings,
        ids=ids,
        method_configs=configs,
        top_k=1,
    )

    backends = {row["backend"] for row in rows}
    assert {"faiss", "scann", "hnswlib"} <= backends


def test_benchmark_methods_collects_hits_returns_details() -> None:
    embeddings = np.eye(2, dtype="float32")
    ids = [f"img-{i}" for i in range(2)]
    ground_truth = [["img-0"], ["img-1"]]
    query_ids = [f"q-{i}" for i in range(2)]
    query_metadata = [{"query_text": "zero"}, {"query_text": "one"}]

    rows, details = benchmark_methods(
        embeddings,
        embeddings,
        ids=ids,
        method_configs=[{"backend": "faiss", "method": "flat", "metric": "l2"}],
        top_k=1,
        ground_truth=ground_truth,
        recall_points=[1],
        collect_hits=True,
        query_ids=query_ids,
        query_metadata=query_metadata,
    )

    assert rows[0]["accuracy"] == 1.0
    assert len(details) == 2
    assert {detail["rank"] for detail in details} == {1}
    assert {detail["query_text"] for detail in details} == {"zero", "one"}
    assert all(detail["is_relevant"] for detail in details)


def test_benchmark_bruteforce_handles_extension_mismatch() -> None:
    embeddings = np.eye(2, dtype="float32")
    ids = [f"img-{i}" for i in range(2)]
    ground_truth = [["img-0.jpg"], ["img-1.JPG"]]

    rows = benchmark_bruteforce(
        embeddings,
        embeddings,
        ids=ids,
        metrics=["l2"],
        top_k=1,
        ground_truth=ground_truth,
        recall_points=[1],
    )

    assert rows[0]["accuracy"] == 1.0
    assert rows[0]["recall@1"] == 1.0
