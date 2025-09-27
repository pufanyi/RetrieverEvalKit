"""Milvus-backed embedding storage helpers.

This module exposes a thin wrapper around a Milvus collection so the rest of
the application can store and query image embeddings without juggling
`pymilvus` primitives directly.  The implementation is intentionally minimal –
enough to bootstrap a local Milvus deployment for development and testing –
while still exposing convenience helpers for inserting, searching, and
deleting embeddings.

Example
-------

```python
from img_search.database.embeddings import EmbeddingDatabase

db = EmbeddingDatabase("img_embeddings", dim=512)
db.add_embeddings(
    ids=["cat-1", "cat-2"],
    embeddings=[cat_vector, another_cat_vector],
    model_name="siglip2",
    dataset_name="cats",
)

results = db.search(cat_query_vector, top_k=5)
```

The class manages connection lifecycle, schema bootstrapping, and index
creation on first use.  Consumers only need a running Milvus instance – e.g.
`milvus-standalone` via Docker – listening on ``host``/``port``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

try:  # Optional dependency when Hydra isn't installed in minimal contexts
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - Hydra-less environments

    class _DictConfigFallback:  # type: ignore[too-many-ancestors]
        pass

    DictConfig = _DictConfigFallback  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

DEFAULT_INDEX_PARAMS: dict[str, Any] = {"M": 16, "efConstruction": 200}
DEFAULT_SEARCH_PARAMS: dict[str, Any] = {"params": {"ef": 64}}


def _as_float_vectors(vectors: Iterable[np.ndarray], *, dim: int) -> list[list[float]]:
    data: list[list[float]] = []
    for vector in vectors:
        array = np.asarray(vector, dtype="float32")
        if array.ndim != 1:
            raise ValueError("Each embedding must be a one-dimensional vector")
        if array.shape[0] != dim:
            raise ValueError(
                "Embedding dimension mismatch: expected "
                f"{dim}, received {array.shape[0]}"
            )
        data.append(array.tolist())
    return data


def _make_expr(field: str, ids: Sequence[str]) -> str:
    quoted = ", ".join(f'"{identifier}"' for identifier in ids)
    return f"{field} in [{quoted}]"


def _config_to_dict(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
    else:
        container = cfg
    if not isinstance(container, Mapping):
        raise TypeError("Embedding database config must be a mapping")
    return {str(key): value for key, value in container.items()}


def create_embedding_database(
    cfg: DictConfig | Mapping[str, Any],
    *,
    dim: int,
) -> EmbeddingDatabase:
    """Instantiate :class:`EmbeddingDatabase` from a Hydra/OmegaConf config."""

    config = _config_to_dict(cfg)

    connection_cfg = config.get("connection", {})
    if connection_cfg is None:
        connection_cfg = {}
    if not isinstance(connection_cfg, Mapping):
        raise TypeError("database.connection must be a mapping when provided")

    index_cfg = config.get("index", {})
    if index_cfg is None:
        index_cfg = {}
    if not isinstance(index_cfg, Mapping):
        raise TypeError("database.index must be a mapping when provided")

    storage_cfg = config.get("storage", {})
    if storage_cfg is None:
        storage_cfg = {}
    if not isinstance(storage_cfg, Mapping):
        raise TypeError("database.storage must be a mapping when provided")

    collection_name = config.get("collection_name", "img_embeddings")
    host = connection_cfg.get("host", "127.0.0.1")
    port = connection_cfg.get("port", 19530)
    alias = connection_cfg.get("alias", "default")
    metric_type = config.get("metric_type", "IP")

    index_type = index_cfg.get("type", config.get("index_type", "HNSW"))
    index_params = index_cfg.get("params", config.get("index_params"))
    if index_params is not None and not isinstance(index_params, Mapping):
        raise TypeError("database.index.params must be a mapping when provided")
    index_params_dict = (
        dict(index_params) if isinstance(index_params, Mapping) else None
    )

    load_on_init = config.get("load_on_init", True)
    storage_path = storage_cfg.get("path", config.get("path"))

    return EmbeddingDatabase(
        collection_name,
        dim,
        host=str(host),
        port=port,
        alias=str(alias),
        metric_type=str(metric_type),
        index_type=str(index_type),
        index_params=index_params_dict,
        load_on_init=bool(load_on_init),
        storage_path=storage_path,
    )


class EmbeddingDatabase:
    """Milvus collection wrapper for storing and querying embeddings."""

    __slots__ = (
        "collection_name",
        "dim",
        "host",
        "port",
        "alias",
        "metric_type",
        "index_type",
        "index_params",
        "load_on_init",
        "storage_path",
        "_collection",
    )

    def __init__(
        self,
        collection_name: str,
        dim: int,
        *,
        host: str = "127.0.0.1",
        port: int | str = 19530,
        alias: str = "default",
        metric_type: str = "IP",
        index_type: str = "HNSW",
        index_params: dict[str, Any] | None = None,
        load_on_init: bool = True,
        storage_path: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self.alias = alias
        self.metric_type = metric_type
        self.index_type = index_type
        self.index_params = index_params
        self.load_on_init = load_on_init
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self._collection: Collection | None = None
        self._connect()
        self._collection = self._ensure_collection()
        if self.load_on_init:
            self._collection.load()

    # ------------------------------------------------------------------
    # Connection / collection lifecycle helpers
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        if not connections.has_connection(self.alias):
            connections.connect(self.alias, host=self.host, port=str(self.port))

    def _build_schema(self) -> CollectionSchema:
        return CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=255,
                ),
                FieldSchema(
                    name="model_name",
                    dtype=DataType.VARCHAR,
                    max_length=255,
                ),
                FieldSchema(
                    name="dataset_name",
                    dtype=DataType.VARCHAR,
                    max_length=255,
                ),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dim,
                ),
            ],
            description="Image embedding store",
        )

    def _ensure_collection(self) -> Collection:
        if not utility.has_collection(self.collection_name, using=self.alias):
            schema = self._build_schema()
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self.alias,
                consistency_level="Strong",
            )
            index_params = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": self.index_params or DEFAULT_INDEX_PARAMS,
            }
            collection.create_index(field_name="vector", index_params=index_params)
            return collection
        return Collection(self.collection_name, using=self.alias)

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            raise RuntimeError("Collection has not been initialized")
        return self._collection

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def add_embeddings(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[np.ndarray],
        model_name: str,
        dataset_name: str,
    ) -> list[str]:
        if len(ids) != len(embeddings):
            raise ValueError("`ids` and `embeddings` must have the same length")

        vectors = _as_float_vectors(embeddings, dim=self.dim)
        self.collection.load()
        try:
            result = self.collection.upsert(
                data=[
                    list(ids),
                    [model_name] * len(ids),
                    [dataset_name] * len(ids),
                    vectors,
                ]
            )
        except AttributeError:
            # Older Milvus releases do not expose ``Collection.upsert``; emulate it.
            self.delete(ids)
            result = self.collection.insert(
                [
                    list(ids),
                    [model_name] * len(ids),
                    [dataset_name] * len(ids),
                    vectors,
                ]
            )
        return list(result.primary_keys)  # type: ignore[no-any-return]

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        expr = _make_expr("id", ids)
        self.collection.delete(expr)

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 10,
        filter_expression: str | None = None,
        output_fields: Sequence[str] | None = ("model_name", "dataset_name"),
        search_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        vector = _as_float_vectors([query], dim=self.dim)[0]
        params = search_params or {
            **DEFAULT_SEARCH_PARAMS,
            "metric_type": self.metric_type,
        }
        self.collection.load()
        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param=params,
            limit=top_k,
            expr=filter_expression,
            output_fields=list(output_fields) if output_fields else None,
        )
        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            payload = {
                "id": hit.id,
                "distance": hit.distance,
            }
            entity = hit.entity
            if entity is not None and output_fields:
                for field in output_fields:
                    payload[field] = entity.get(field)
            hits.append(payload)
        return hits

    def drop(self) -> None:
        try:
            utility.drop_collection(self.collection_name, using=self.alias)
        except MilvusException:
            # Collection might have already been removed; ignore.
            pass

    def flush(self) -> None:
        self.collection.flush()

    def close(self) -> None:
        try:
            self.collection.release()
        finally:
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)

    @contextmanager
    def session(self) -> Iterator[Collection]:
        try:
            yield self.collection
        finally:
            self.flush()


__all__ = ["EmbeddingDatabase", "create_embedding_database"]
