"""FAISS-based vector search utilities.

This module centralises FAISS index construction so experiments can compare
multiple ANN strategies and measure their trade-offs.  The
:class:`FaissSearchIndex` helper wraps common index types (flat, IVF, PQ and
HNSW) and exposes a uniform interface for inserting embeddings and performing
nearest-neighbour queries.  Callers can optionally mirror the index on GPU to
benchmark CPU vs GPU throughput.

The accompanying :func:`benchmark_methods` function streamlines evaluating a set
of index configurations by collecting timing metrics and (optionally) top-k
accuracy when ground-truth neighbours are provided.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Mapping, Sequence, Literal

import numpy as np

try:  # pragma: no cover - exercised in tests when faiss is available
    import faiss  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "FAISS is required for img_search.search. Install `faiss-cpu` or `faiss-gpu`."
    ) from exc

@dataclass(slots=True)
class FaissIndexConfig:
    """Configuration describing how the FAISS index should be built."""

    method: Literal["flat", "ivf_flat", "ivf_pq", "hnsw"] = "flat"
    metric: Literal["l2", "ip", "cosine"] = "l2"
    nlist: int = 1024
    nprobe: int = 32
    m: int = 8
    nbits: int = 8
    hnsw_m: int = 32
    ef_search: int = 64
    normalize: bool | None = None
    use_gpu: bool = False
    gpu_device: int | None = None
    temp_memory: int | None = None

    def __post_init__(self) -> None:
        self.method = str(self.method).lower()
        if self.method not in {"flat", "ivf_flat", "ivf_pq", "hnsw"}:
            raise ValueError(
                "Unsupported FAISS method. Choose from 'flat', 'ivf_flat',"
                " 'ivf_pq', or 'hnsw'."
            )

        self.metric = str(self.metric).lower()
        if self.metric not in {"l2", "ip", "cosine"}:
            raise ValueError("metric must be 'l2', 'ip', or 'cosine'")

        self.nlist = int(self.nlist)
        if self.nlist <= 0:
            raise ValueError("nlist must be > 0")

        self.nprobe = int(self.nprobe)
        if self.nprobe <= 0:
            raise ValueError("nprobe must be > 0")
        if self.nprobe > self.nlist:
            raise ValueError("nprobe cannot exceed nlist")

        self.m = int(self.m)
        if self.m <= 0:
            raise ValueError("m must be > 0")

        self.nbits = int(self.nbits)
        if self.nbits <= 0:
            raise ValueError("nbits must be > 0")

        self.hnsw_m = int(self.hnsw_m)
        if self.hnsw_m <= 0:
            raise ValueError("hnsw_m must be > 0")

        self.ef_search = int(self.ef_search)
        if self.ef_search <= 0:
            raise ValueError("ef_search must be > 0")

        if self.normalize is None:
            self.normalize = self.metric == "cosine"

        if self.use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("FAISS was compiled without GPU support")
            available = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
            if available == 0:
                raise RuntimeError("No GPU devices available for FAISS")
            if self.gpu_device is None:
                self.gpu_device = 0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FaissIndexConfig":
        if mapping is None:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{key: mapping[key] for key in valid_keys if key in mapping})


class FaissSearchIndex:
    """Wrapper around FAISS indexes supporting multiple search strategies."""

    def __init__(self, dim: int, *, config: FaissIndexConfig | Mapping[str, Any] | None = None) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be > 0")

        if isinstance(config, Mapping):
            self.config = FaissIndexConfig.from_mapping(config)
        elif isinstance(config, FaissIndexConfig) or config is None:
            self.config = config or FaissIndexConfig()
        else:  # pragma: no cover - defensive guard
            raise TypeError("config must be a mapping or FaissIndexConfig")

        self._cpu_index = self._build_index()
        self._gpu_index: faiss.Index | None = None
        self._gpu_resources: faiss.StandardGpuResources | None = None
        self._gpu_dirty = False

        self._id_lookup: dict[int, str] = {}
        self._label_lookup: dict[str, int] = {}
        self._next_id = 0

        if self.config.use_gpu:
            self._gpu_resources = faiss.StandardGpuResources()
            if self.config.temp_memory is not None:
                self._gpu_resources.setTempMemory(self.config.temp_memory)
            self._gpu_dirty = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def ntotal(self) -> int:
        return int(self._cpu_index.ntotal)

    def reset(self) -> None:
        self._cpu_index.reset()
        self._gpu_index = None
        self._gpu_dirty = self.config.use_gpu
        self._id_lookup.clear()
        self._label_lookup.clear()
        self._next_id = 0

    def add_embeddings(self, ids: Sequence[str], embeddings: Iterable[np.ndarray | Sequence[float]]) -> None:
        ids = list(ids)
        vectors = _as_matrix(embeddings, dim=self.dim)
        if len(ids) != len(vectors):
            raise ValueError("ids and embeddings must have the same length")

        if self.config.normalize:
            faiss.normalize_L2(vectors)

        if not self._cpu_index.is_trained:
            self._cpu_index.train(vectors)

        faiss_ids = np.empty(len(ids), dtype="int64")
        for idx, label in enumerate(ids):
            if label in self._label_lookup:
                raise ValueError(f"Duplicate identifier detected: {label}")
            faiss_id = self._next_id
            self._next_id += 1
            self._label_lookup[label] = faiss_id
            self._id_lookup[faiss_id] = label
            faiss_ids[idx] = faiss_id

        self._cpu_index.add_with_ids(vectors, faiss_ids)
        self._gpu_dirty = self.config.use_gpu

    def search(
        self,
        query: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        top_k: int = 10,
        return_time: bool = False,
    ) -> Any:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        queries = _as_matrix(query, dim=self.dim)
        if self.config.normalize:
            faiss.normalize_L2(queries)

        index = self._active_index()
        start = time.perf_counter()
        distances, ids = index.search(queries, top_k)
        elapsed = time.perf_counter() - start

        results: list[list[dict[str, Any]]] = []
        for row_distances, row_ids in zip(distances, ids, strict=True):
            row: list[dict[str, Any]] = []
            for distance, identifier in zip(row_distances, row_ids, strict=True):
                if identifier < 0:
                    continue
                label = self._id_lookup.get(int(identifier))
                if label is None:
                    continue
                row.append({"id": label, "distance": float(distance)})
            results.append(row)

        if queries.shape[0] == 1:
            single_result: list[dict[str, Any]] = results[0] if results else []
            if return_time:
                return single_result, elapsed
            return single_result

        if return_time:
            return results, elapsed
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_index(self) -> faiss.Index:
        metric_type = _metric_type(self.config.metric)

        if self.config.method == "flat":
            base_index = _flat_index(self.dim, metric_type)
        elif self.config.method == "ivf_flat":
            quantizer = _flat_index(self.dim, metric_type)
            base_index = faiss.IndexIVFFlat(quantizer, self.dim, self.config.nlist, metric_type)
            base_index.nprobe = self.config.nprobe
        elif self.config.method == "ivf_pq":
            quantizer = _flat_index(self.dim, metric_type)
            base_index = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                self.config.nlist,
                self.config.m,
                self.config.nbits,
                metric_type,
            )
            base_index.nprobe = self.config.nprobe
        else:  # self.config.method == "hnsw"
            base_index = faiss.IndexHNSWFlat(self.dim, self.config.hnsw_m)
            base_index.hnsw.efConstruction = max(
                self.config.hnsw_m * 2, self.config.ef_search
            )
            base_index.hnsw.efSearch = self.config.ef_search
            base_index.metric_type = metric_type

        return faiss.IndexIDMap2(base_index)

    def _active_index(self) -> faiss.Index:
        if not self.config.use_gpu:
            return self._cpu_index
        if self._gpu_dirty or self._gpu_index is None:
            assert self._gpu_resources is not None  # for type checkers
            self._gpu_index = faiss.index_cpu_to_gpu(
                self._gpu_resources, int(self.config.gpu_device), self._cpu_index
            )
            self._gpu_dirty = False
        return self._gpu_index


def benchmark_methods(
    embeddings: Iterable[np.ndarray | Sequence[float]],
    queries: Iterable[np.ndarray | Sequence[float]],
    *,
    ids: Sequence[str] | None = None,
    method_configs: Sequence[FaissIndexConfig | Mapping[str, Any]],
    top_k: int = 5,
    ground_truth: Sequence[Sequence[str] | str] | None = None,
    recall_points: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Benchmark multiple FAISS configurations on shared data."""

    embeddings_matrix = _as_matrix(embeddings)
    query_matrix = _as_matrix(queries, dim=embeddings_matrix.shape[1])
    if ids is None:
        ids = [str(idx) for idx in range(len(embeddings_matrix))]
    if len(ids) != len(embeddings_matrix):
        raise ValueError("ids length must match embeddings length")

    results: list[dict[str, Any]] = []

    recall_points = sorted({int(point) for point in recall_points or [] if int(point) > 0})

    for cfg in method_configs:
        index_config = (
            FaissIndexConfig.from_mapping(cfg) if isinstance(cfg, Mapping) else cfg
        )
        index = FaissSearchIndex(embeddings_matrix.shape[1], config=index_config)

        build_start = time.perf_counter()
        index.add_embeddings(ids, embeddings_matrix)
        build_time = time.perf_counter() - build_start

        search_start = time.perf_counter()
        hits = index.search(query_matrix, top_k=top_k)
        search_time = time.perf_counter() - search_start

        if not hits:
            per_query_results = [[] for _ in range(query_matrix.shape[0])]
        elif isinstance(hits[0], dict):
            per_query_results = [hits]  # type: ignore[list-item]
        else:
            per_query_results = hits  # type: ignore[assignment]

        accuracy = None
        recall_totals: dict[int, float] = {k: 0.0 for k in recall_points}
        recall_counts: dict[int, int] = {k: 0 for k in recall_points}
        if ground_truth is not None:
            if len(ground_truth) != len(per_query_results):
                raise ValueError("ground_truth length must match number of queries")
            correct = 0
            for expected, retrieved in zip(ground_truth, per_query_results, strict=True):
                expected_ids = _as_label_set(expected)
                if any(hit["id"] in expected_ids for hit in retrieved):
                    correct += 1
                if recall_points:
                    if not expected_ids:
                        continue
                    relevant_count = len(expected_ids)
                    retrieved_ids = [hit["id"] for hit in retrieved]
                    for point in recall_points:
                        if point > len(retrieved_ids):
                            subset = retrieved_ids
                        else:
                            subset = retrieved_ids[:point]
                        hits = len(expected_ids.intersection(subset))
                        recall_totals[point] += hits / relevant_count
                        recall_counts[point] += 1
            accuracy = correct / len(per_query_results)

        row: dict[str, Any] = {
            "method": index_config.method,
            "metric": index_config.metric,
            "use_gpu": index_config.use_gpu,
            "index_time": build_time,
            "search_time": search_time,
            "avg_query_time": search_time / len(per_query_results),
            "accuracy": accuracy,
            "ntotal": index.ntotal,
        }
        for point in recall_points:
            if recall_counts[point] == 0:
                row[f"recall@{point}"] = None
            else:
                row[f"recall@{point}"] = recall_totals[point] / recall_counts[point]

        results.append(row)

    return results


def _as_label_set(values: Sequence[str] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values}
    result = {str(item) for item in values}
    return result


def _metric_type(metric: str) -> int:
    metric = metric.lower()
    if metric == "l2":
        return faiss.METRIC_L2
    if metric in {"ip", "cosine"}:
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"Unsupported metric: {metric}")


def _flat_index(dim: int, metric_type: int) -> faiss.Index:
    if metric_type == faiss.METRIC_L2:
        return faiss.IndexFlatL2(dim)
    return faiss.IndexFlatIP(dim)


def _as_matrix(
    data: Iterable[np.ndarray | Sequence[float]],
    *,
    dim: int | None = None,
) -> np.ndarray:
    if isinstance(data, np.ndarray):
        array = np.asarray(data, dtype="float32")
    else:
        array = np.asarray(list(data), dtype="float32")
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 2:
        raise ValueError("Expected 2D array of embeddings")
    if dim is not None and array.shape[1] != dim:
        raise ValueError(f"Expected vectors of dimension {dim}, received {array.shape[1]}")
    return array


__all__ = ["FaissIndexConfig", "FaissSearchIndex", "benchmark_methods"]
