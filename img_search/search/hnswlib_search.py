"""HNSWlib-based vector search utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import hnswlib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hnswlib = None  # type: ignore[assignment]


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
class HnswlibIndexConfig:
    """Configuration describing how the HNSWlib index should be built."""

    metric: str = "l2"
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    normalise: bool | None = None

    def __post_init__(self) -> None:
        self.metric = str(self.metric).lower()
        if self.metric not in {"l2", "ip", "cosine"}:
            raise ValueError("metric must be 'l2', 'ip', or 'cosine'")

        self.m = int(self.m)
        if self.m <= 0:
            raise ValueError("m must be > 0")

        self.ef_construction = int(self.ef_construction)
        if self.ef_construction <= 0:
            raise ValueError("ef_construction must be > 0")

        self.ef_search = int(self.ef_search)
        if self.ef_search <= 0:
            raise ValueError("ef_search must be > 0")

        if self.normalise is None:
            self.normalise = self.metric == "cosine"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> HnswlibIndexConfig:
        valid_keys = cls.__dataclass_fields__.keys()
        kwargs = {key: mapping[key] for key in valid_keys if key in mapping}
        return cls(**kwargs)


class HnswlibSearchIndex:
    """Wrapper around a HNSWlib index exposing a FAISS-like interface."""

    def __init__(
        self, dim: int, *, config: HnswlibIndexConfig | Mapping[str, Any] | None = None
    ) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be > 0")

        if isinstance(config, Mapping):
            self.config = HnswlibIndexConfig.from_mapping(config)
        elif isinstance(config, HnswlibIndexConfig) or config is None:
            self.config = config or HnswlibIndexConfig()
        else:  # pragma: no cover - defensive guard
            raise TypeError("config must be a mapping or HnswlibIndexConfig")

        self._index: Any | None = None
        self._ids: list[str] = []

    @property
    def ntotal(self) -> int:
        return len(self._ids)

    def _require_index(self) -> None:
        if hnswlib is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "HNSWlib is required for HnswlibSearchIndex. "
                "Install the 'hnswlib' package."
            )

    def _space(self) -> str:
        if self.config.metric == "l2":
            return "l2"
        return "ip"

    def add_embeddings(
        self, ids: Sequence[str], embeddings: Iterable[np.ndarray | Sequence[float]]
    ) -> None:
        self._require_index()
        matrix = _as_matrix(embeddings)
        if matrix.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of dimension {self.dim}, received {matrix.shape[1]}"
            )
        if self.config.metric == "cosine" and self.config.normalise:
            matrix = _normalise_rows(matrix)

        self._ids = [str(identifier) for identifier in ids]
        if len(self._ids) != len(matrix):
            raise ValueError("ids length must match embeddings length")

        self._index = hnswlib.Index(space=self._space(), dim=self.dim)
        self._index.init_index(
            max_elements=len(matrix),
            ef_construction=self.config.ef_construction,
            M=self.config.m,
        )
        labels = np.arange(len(matrix), dtype=np.int64)
        self._index.add_items(matrix, labels)
        self._index.set_ef(self.config.ef_search)

    def search(
        self, queries: Iterable[np.ndarray | Sequence[float]], *, top_k: int
    ) -> list[list[dict[str, float | str]]]:
        if self._index is None:
            raise RuntimeError("Index has not been built. Call add_embeddings first")

        query_matrix = _as_matrix(queries)
        if query_matrix.shape[1] != self.dim:
            raise ValueError(
                f"Expected query dimension {self.dim}, received {query_matrix.shape[1]}"
            )

        if self.config.metric == "cosine" and self.config.normalise:
            query_matrix = _normalise_rows(query_matrix)

        actual_k = min(int(top_k), len(self._ids))
        if actual_k <= 0:
            raise ValueError("top_k must be a positive integer")

        labels, distances = self._index.knn_query(query_matrix, k=actual_k)
        results: list[list[dict[str, float | str]]] = []
        for row_labels, row_distances in zip(labels, distances, strict=True):
            row: list[dict[str, float | str]] = []
            for label, distance in zip(row_labels, row_distances, strict=True):
                if int(label) < 0:
                    continue
                row.append({"id": self._ids[int(label)], "distance": float(distance)})
            results.append(row)
        return results


def hnswlib_available() -> bool:
    """Return ``True`` when the HNSWlib dependency can be imported."""

    return hnswlib is not None


__all__ = ["HnswlibIndexConfig", "HnswlibSearchIndex", "hnswlib_available"]
