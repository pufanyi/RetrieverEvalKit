from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from img_search.database import embeddings as embeddings_module
from img_search.database.embeddings import (
    EmbeddingDatabase,
    _as_float_vectors,
    create_embedding_database,
)

np = pytest.importorskip("numpy")


class DummyFieldSchema:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class DummyCollectionSchema:
    def __init__(self, *, fields: list[Any], description: str | None = None) -> None:
        self.fields = fields
        self.description = description


class DummyDataType:
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"


class FakeConnections:
    def __init__(self) -> None:
        self.connected: dict[str, tuple[str, str]] = {}

    def has_connection(self, alias: str) -> bool:
        return alias in self.connected

    def connect(self, alias: str, host: str, port: str) -> None:
        self.connected[alias] = (host, port)

    def disconnect(self, alias: str) -> None:
        self.connected.pop(alias, None)


class FakeUtility:
    def has_collection(self, name: str, *, using: str | None = None) -> bool:  # noqa: ARG002
        return name in FakeCollection.registry

    def drop_collection(self, name: str, *, using: str | None = None) -> None:  # noqa: ARG002
        FakeCollection.registry.pop(name, None)


class FakeHit:
    def __init__(self, id_: str, distance: float, fields: dict[str, Any]) -> None:
        self.id = id_
        self.distance = distance
        self.entity = fields


class FakeCollection:
    registry: dict[str, FakeCollection] = {}

    def __new__(cls, name: str, *args: Any, **kwargs: Any) -> FakeCollection:
        if kwargs.get("schema") is not None:
            instance = super().__new__(cls)
            cls.registry[name] = instance
            return instance
        if name in cls.registry:
            return cls.registry[name]
        instance = super().__new__(cls)
        cls.registry[name] = instance
        return instance

    def __init__(
        self,
        name: str,
        schema: DummyCollectionSchema | None = None,
        *,
        using: str | None = None,
        consistency_level: str | None = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        self.name = name
        self.schema = schema
        self.using = using
        self.consistency_level = consistency_level
        self.loaded = False
        self.indexes: list[tuple[str, dict[str, Any]]] = []
        self.upserts: list[list[Any]] = []
        self.inserts: list[list[Any]] = []
        self.deleted_exprs: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.search_results: list[list[FakeHit]] = [[]]
        self.flushed = False
        self.released = False
        self._initialized = True

    def create_index(self, *, field_name: str, index_params: dict[str, Any]) -> None:
        self.indexes.append((field_name, index_params))

    def load(self) -> None:
        self.loaded = True

    def upsert(self, data: list[Any]) -> SimpleNamespace:
        self.upserts.append(data)
        return SimpleNamespace(primary_keys=data[0])

    def insert(self, data: list[Any]) -> SimpleNamespace:
        self.inserts.append(data)
        return SimpleNamespace(primary_keys=data[0])

    def delete(self, expr: str) -> None:
        self.deleted_exprs.append(expr)

    def search(
        self,
        *,
        data: list[list[float]],
        anns_field: str,
        param: dict[str, Any],
        limit: int,
        expr: str | None,
        output_fields: list[str] | None,
    ) -> list[list[FakeHit]]:
        self.search_calls.append(
            {
                "data": data,
                "anns_field": anns_field,
                "param": param,
                "limit": limit,
                "expr": expr,
                "output_fields": output_fields,
            }
        )
        return self.search_results

    def flush(self) -> None:
        self.flushed = True

    def release(self) -> None:
        self.released = True


@pytest.fixture(autouse=True)
def mock_milvus(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    FakeCollection.registry.clear()
    fake_connections = FakeConnections()
    fake_utility = FakeUtility()

    monkeypatch.setattr(embeddings_module, "connections", fake_connections)
    monkeypatch.setattr(embeddings_module, "utility", fake_utility)
    monkeypatch.setattr(embeddings_module, "Collection", FakeCollection)
    monkeypatch.setattr(embeddings_module, "CollectionSchema", DummyCollectionSchema)
    monkeypatch.setattr(embeddings_module, "FieldSchema", DummyFieldSchema)
    monkeypatch.setattr(embeddings_module, "DataType", DummyDataType)
    monkeypatch.setattr(embeddings_module, "MilvusException", RuntimeError)
    return {"connections": fake_connections, "utility": fake_utility}


def test_collection_initialization_creates_index() -> None:
    db = EmbeddingDatabase("test_collection", dim=4)

    collection = FakeCollection.registry["test_collection"]
    assert collection.schema is not None
    assert collection.loaded is True
    assert collection.indexes == [
        (
            "vector",
            {
                "index_type": db.index_type,
                "metric_type": db.metric_type,
                "params": embeddings_module.DEFAULT_INDEX_PARAMS,
            },
        )
    ]


def test_create_embedding_database_from_config() -> None:
    cfg = {
        "collection_name": "cfg_collection",
        "connection": {"host": "milvus", "port": 19531, "alias": "cfg"},
        "metric_type": "L2",
        "index": {"type": "DISKANN", "params": {"search_list": 50}},
        "storage": {"path": "database/embeddings.db"},
        "load_on_init": False,
    }

    db = create_embedding_database(cfg, dim=8)

    assert db.collection_name == "cfg_collection"
    assert db.host == "milvus"
    assert db.port == 19531
    assert db.alias == "cfg"
    assert db.metric_type == "L2"
    assert db.index_type == "DISKANN"
    assert db.index_params == {"search_list": 50}
    assert db.storage_path == Path("database/embeddings.db")
    assert db.dim == 8
    collection = FakeCollection.registry["cfg_collection"]
    assert collection.loaded is False
    assert collection.indexes == [
        (
            "vector",
            {
                "index_type": "DISKANN",
                "metric_type": "L2",
                "params": {"search_list": 50},
            },
        )
    ]


def test_add_embeddings_uses_upsert_and_returns_ids() -> None:
    db = EmbeddingDatabase("demo", dim=2)
    vectors = [
        np.array([1.0, 0.0], dtype="float32"),
        np.array([0.0, 1.0], dtype="float32"),
    ]

    ids = db.add_embeddings(
        ids=["a", "b"],
        embeddings=vectors,
        model_name="model",
        dataset_name="dataset",
    )

    assert ids == ["a", "b"]
    collection = FakeCollection.registry["demo"]
    assert collection.upserts == [
        [
            ["a", "b"],
            ["model", "model"],
            ["dataset", "dataset"],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    ]


def test_search_returns_hit_payload() -> None:
    db = EmbeddingDatabase("search", dim=3)
    collection = FakeCollection.registry["search"]
    collection.search_results = [
        [
            FakeHit("item-1", 0.1, {"model_name": "m1", "dataset_name": "d1"}),
            FakeHit("item-2", 0.2, {"model_name": "m1", "dataset_name": "d2"}),
        ]
    ]

    results = db.search(np.array([1.0, 2.0, 3.0], dtype="float32"), top_k=2)

    assert results == [
        {"id": "item-1", "distance": 0.1, "model_name": "m1", "dataset_name": "d1"},
        {"id": "item-2", "distance": 0.2, "model_name": "m1", "dataset_name": "d2"},
    ]
    assert collection.search_calls[0]["limit"] == 2
    assert collection.search_calls[0]["anns_field"] == "vector"


def test_delete_builds_expression() -> None:
    db = EmbeddingDatabase("delete", dim=2)
    collection = FakeCollection.registry["delete"]

    db.delete(["x", "y"])

    assert collection.deleted_exprs == ['id in ["x", "y"]']


def test_drop_and_close_release_resources() -> None:
    db = EmbeddingDatabase("cleanup", dim=2)
    collection = FakeCollection.registry["cleanup"]

    db.drop()
    assert "cleanup" not in FakeCollection.registry

    db.close()
    assert collection.released is True


def test_as_float_vectors_validates_shape() -> None:
    with pytest.raises(ValueError):
        _as_float_vectors([np.zeros((2, 2))], dim=4)

    with pytest.raises(ValueError):
        _as_float_vectors([np.zeros(3)], dim=4)
