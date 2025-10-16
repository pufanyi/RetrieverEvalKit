import numpy as np

from img_search.search.faiss_search import benchmark_methods
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
