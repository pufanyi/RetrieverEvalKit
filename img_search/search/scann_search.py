"""ScaNN-based vector search utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import scann  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    scann = None  # type: ignore[assignment]


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _as_matrix(data: Iterable[np.ndarray | Sequence[float]]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        array = np.asarray(data, dtype="float32")
    else:
        array = np.asarray(list(data), dtype="float32")
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 2:
        raise ValueError("Expected 2D array of embeddings")
    return array


@dataclass(slots=True)
class ScannIndexConfig:
    """Configuration describing how the ScaNN index should be built."""

    metric: str = "l2"
    num_leaves: int | None = None
    num_leaves_to_search: int | None = None
    num_neighbors: int = 10
    reorder_k: int | None = None
    normalise: bool | None = None

    def __post_init__(self) -> None:
        self.metric = str(self.metric).lower()
        if self.metric not in {"l2", "ip", "cosine"}:
            raise ValueError("metric must be 'l2', 'ip', or 'cosine'")

        if self.num_leaves is not None:
            self.num_leaves = int(self.num_leaves)
            if self.num_leaves <= 0:
                raise ValueError("num_leaves must be > 0")
        if self.num_leaves_to_search is not None:
            if self.num_leaves is None:
                raise ValueError(
                    "num_leaves_to_search requires num_leaves to be configured"
                )
            self.num_leaves_to_search = int(self.num_leaves_to_search)
            if self.num_leaves_to_search <= 0:
                raise ValueError("num_leaves_to_search must be > 0")
            if self.num_leaves_to_search > self.num_leaves:
                self.num_leaves_to_search = self.num_leaves

        self.num_neighbors = int(self.num_neighbors)
        if self.num_neighbors <= 0:
            raise ValueError("num_neighbors must be > 0")

        if self.reorder_k is not None:
            self.reorder_k = int(self.reorder_k)
            if self.reorder_k <= 0:
                raise ValueError("reorder_k must be > 0 when provided")

        if self.normalise is None:
            self.normalise = self.metric == "cosine"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ScannIndexConfig:
        valid_keys = cls.__dataclass_fields__.keys()
        kwargs = {key: mapping[key] for key in valid_keys if key in mapping}
        return cls(**kwargs)


class ScannSearchIndex:
    """Wrapper around a ScaNN searcher exposing a FAISS-like interface."""

    def __init__(
        self, dim: int, *, config: ScannIndexConfig | Mapping[str, Any] | None = None
    ) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be > 0")

        if isinstance(config, Mapping):
            self.config = ScannIndexConfig.from_mapping(config)
        elif isinstance(config, ScannIndexConfig) or config is None:
            self.config = config or ScannIndexConfig()
        else:  # pragma: no cover - defensive guard
            raise TypeError("config must be a mapping or ScannIndexConfig")

        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._searcher: Any | None = None
        self._search_k = self.config.num_neighbors
        self._normalise_queries = bool(self.config.normalise)

    @property
    def ntotal(self) -> int:
        return len(self._ids)

    def _require_searcher(self) -> None:
        if scann is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ScaNN is required for ScannSearchIndex. Install the 'scann' package."
            )
        if self._embeddings is None:
            raise RuntimeError("Embeddings must be added before building the index")

    def _metric_name(self) -> str:
        if self.config.metric == "l2":
            return "squared_l2"
        return "dot_product"

    def _prepare_embeddings(self) -> np.ndarray:
        assert self._embeddings is not None
        if self.config.metric == "cosine" and self.config.normalise:
            return _normalise_rows(self._embeddings)
        return self._embeddings

    def _build_searcher(self, *, search_neighbors: int) -> None:
        self._require_searcher()
        assert self._embeddings is not None

        dataset = self._prepare_embeddings()
        builder = scann.scann_ops_pybind.builder(
            dataset,
            int(search_neighbors),
            self._metric_name(),
        )

        if self.config.num_leaves is not None:
            leaves = int(self.config.num_leaves)
            leaves_to_search = (
                int(self.config.num_leaves_to_search)
                if self.config.num_leaves_to_search is not None
                else leaves
            )
            builder = builder.tree(
                num_leaves=leaves,
                num_leaves_to_search=leaves_to_search,
            )
        builder = builder.score_brute_force()
        if self.config.reorder_k:
            builder = builder.reorder(int(self.config.reorder_k))

        self._searcher = builder.build()
        self._search_k = int(search_neighbors)

    def add_embeddings(
        self, ids: Sequence[str], embeddings: Iterable[np.ndarray | Sequence[float]]
    ) -> None:
        matrix = _as_matrix(embeddings)
        if matrix.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of dimension {self.dim}, received {matrix.shape[1]}"
            )
        self._ids = [str(identifier) for identifier in ids]
        if len(self._ids) != len(matrix):
            raise ValueError("ids length must match embeddings length")
        self._embeddings = matrix
        self._build_searcher(search_neighbors=self.config.num_neighbors)

    def search(
        self, queries: Iterable[np.ndarray | Sequence[float]], *, top_k: int
    ) -> list[list[dict[str, float | str]]]:
        if self._searcher is None:
            raise RuntimeError("Index has not been built. Call add_embeddings first")

        query_matrix = _as_matrix(queries)
        if query_matrix.shape[1] != self.dim:
            raise ValueError(
                f"Expected query dimension {self.dim}, received {query_matrix.shape[1]}"
            )

        if self._normalise_queries and self.config.metric == "cosine":
            query_matrix = _normalise_rows(query_matrix)

        actual_k = min(int(top_k), len(self._ids))
        if actual_k <= 0:
            raise ValueError("top_k must be a positive integer")

        search_neighbors = max(actual_k, self._search_k)
        if search_neighbors != self._search_k:
            self._build_searcher(search_neighbors=search_neighbors)

        distances, indices = self._searcher.search_batched(
            query_matrix, final_num_neighbors=search_neighbors
        )
        results: list[list[dict[str, float | str]]] = []
        for row_indices, row_distances in zip(indices, distances, strict=True):
            row_indices = row_indices[:actual_k]
            row_distances = row_distances[:actual_k]
            row: list[dict[str, float | str]] = []
            for idx, distance in zip(row_indices, row_distances, strict=True):
                if int(idx) < 0:
                    continue
                row.append({"id": self._ids[int(idx)], "distance": float(distance)})
            results.append(row)
        return results


def scann_available() -> bool:
    """Return ``True`` when the ScaNN dependency can be imported."""

    return scann is not None


__all__ = ["ScannIndexConfig", "ScannSearchIndex", "scann_available"]
