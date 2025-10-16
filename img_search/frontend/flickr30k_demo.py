from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from img_search.data.embeddings import (
    EmbeddingDatasetSpec,
    extract_embeddings,
    load_embedding_dataset,
)
from img_search.embedding.jina_v4 import JinaV4Encoder
from img_search.search.faiss_search import FaissSearchIndex
from img_search.search.hnswlib_search import (
    HnswlibIndexConfig,
    HnswlibSearchIndex,
    hnswlib_available,
)
from img_search.search.scann_search import (
    ScannIndexConfig,
    ScannSearchIndex,
    scann_available,
)

_DEFAULT_METHODS: tuple[dict[str, Any], ...] = (
    {
        "id": "faiss_flat",
        "label": "FAISS Flat (cosine)",
        "backend": "faiss",
        "config": {"method": "flat", "metric": "cosine"},
    },
    {
        "id": "faiss_ivf_flat",
        "label": "FAISS IVF-Flat (cosine)",
        "backend": "faiss",
        "config": {
            "method": "ivf_flat",
            "metric": "cosine",
            "nlist": 2048,
            "nprobe": 64,
        },
    },
    {
        "id": "faiss_hnsw",
        "label": "FAISS HNSW (cosine)",
        "backend": "faiss",
        "config": {
            "method": "hnsw",
            "metric": "cosine",
            "hnsw_m": 32,
            "ef_search": 128,
        },
    },
    {
        "id": "scann_default",
        "label": "ScaNN Tree+Score (cosine)",
        "backend": "scann",
        "config": {
            "metric": "cosine",
            "num_neighbors": 100,
            "num_leaves": 2000,
            "num_leaves_to_search": 100,
            "reorder_k": 250,
        },
    },
    {
        "id": "hnswlib_default",
        "label": "HNSWlib (cosine)",
        "backend": "hnswlib",
        "config": {
            "metric": "cosine",
            "m": 32,
            "ef_construction": 200,
            "ef_search": 128,
        },
    },
)


@dataclass(slots=True)
class DemoSettings:
    image_dataset: EmbeddingDatasetSpec | None = None
    caption_dataset: EmbeddingDatasetSpec | None = None
    image_root: Path | None = None
    image_pattern: str = "{id}.jpg"
    default_top_k: int = 9
    max_top_k: int = 50

    def __post_init__(self) -> None:
        if self.image_dataset is None:
            self.image_dataset = EmbeddingDatasetSpec(
                dataset_name="pufanyi/flickr30k-jina-embeddings-v4",
                dataset_config="images",
                split="train",
                id_column="id",
                embedding_column="embedding",
            )
        if self.caption_dataset is None:
            caption_config = os.getenv("FLICKR30K_CAPTION_CONFIG", "text")
            self.caption_dataset = EmbeddingDatasetSpec(
                dataset_name="pufanyi/flickr30k-jina-embeddings-v4",
                dataset_config=caption_config,
                split="train",
                id_column="id",
                embedding_column="embedding",
            )
        env_root = os.getenv("FLICKR30K_IMAGE_ROOT")
        if self.image_root is None:
            if env_root:
                self.image_root = Path(env_root)
            else:
                self.image_root = Path("data/flickr30k/images")
        else:
            self.image_root = Path(self.image_root)
        env_pattern = os.getenv("FLICKR30K_IMAGE_PATTERN")
        if env_pattern:
            self.image_pattern = env_pattern


@dataclass(slots=True)
class CaptionRecord:
    id: str
    image_id: str | None
    caption: str | None


@dataclass(slots=True)
class BackendInfo:
    identifier: str
    label: str
    backend: str
    metric: str
    config: Mapping[str, Any]
    searcher: Any
    supports_similarity: bool


class Flickr30kSearchEngine:
    def __init__(self, settings: DemoSettings) -> None:
        self.settings = settings
        self._ready = False
        self._build_lock = threading.Lock()
        self._encoder_lock = threading.Lock()
        self._encoder: JinaV4Encoder | None = None

        self._image_ids: list[str] = []
        self._image_vectors: np.ndarray | None = None
        self._image_lookup: dict[str, int] = {}
        self._image_captions: dict[str, list[CaptionRecord]] = {}

        self._caption_vectors: np.ndarray | None = None
        self._caption_lookup: dict[str, int] = {}
        self._caption_metadata: dict[str, CaptionRecord] = {}
        self._caption_samples: list[CaptionRecord] = []

        self._backends: dict[str, BackendInfo] = {}
        self._backend_errors: dict[str, str] = {}

        self.serve_images = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def prepare(self) -> None:
        if self._ready:
            return
        with self._build_lock:
            if self._ready:
                return

            logger.info(
                "Loading Flickr30k image embeddings: %s",
                self.settings.image_dataset,
            )
            image_dataset = load_embedding_dataset(self.settings.image_dataset)
            image_ids, image_vectors = extract_embeddings(
                image_dataset,
                id_column=self.settings.image_dataset.id_column,
                embedding_column=self.settings.image_dataset.embedding_column,
            )
            if image_vectors.ndim != 2:
                raise RuntimeError("Image embedding matrix must be 2-dimensional")
            self._image_ids = [str(identifier) for identifier in image_ids]
            self._image_vectors = np.asarray(image_vectors, dtype="float32")
            self._image_lookup = {
                identifier: index for index, identifier in enumerate(self._image_ids)
            }
            logger.info(
                "Loaded %s image embeddings (dim=%s)",
                len(self._image_ids),
                self._image_vectors.shape[1],
            )

            logger.info(
                "Loading Flickr30k caption embeddings: %s",
                self.settings.caption_dataset,
            )
            caption_dataset = load_embedding_dataset(self.settings.caption_dataset)
            caption_ids, caption_vectors = extract_embeddings(
                caption_dataset,
                id_column=self.settings.caption_dataset.id_column,
                embedding_column=self.settings.caption_dataset.embedding_column,
            )
            if caption_vectors.ndim != 2:
                raise RuntimeError("Caption embedding matrix must be 2-dimensional")
            self._caption_vectors = np.asarray(caption_vectors, dtype="float32")
            self._caption_lookup = {
                str(identifier): index for index, identifier in enumerate(caption_ids)
            }
            logger.info(
                "Loaded %s caption embeddings (dim=%s)",
                len(self._caption_lookup),
                self._caption_vectors.shape[1],
            )

            self._caption_metadata.clear()
            self._image_captions.clear()
            for row in caption_dataset:
                raw_id = row.get(self.settings.caption_dataset.id_column)
                if raw_id is None:
                    continue
                caption_id = str(raw_id)
                image_id_value = row.get("image_id")
                image_id = str(image_id_value) if image_id_value is not None else None
                caption_text_value = row.get("caption")
                caption_text = (
                    None
                    if caption_text_value is None
                    else str(caption_text_value)
                )
                record = CaptionRecord(
                    id=caption_id,
                    image_id=image_id,
                    caption=caption_text,
                )
                self._caption_metadata[caption_id] = record
                if image_id:
                    bucket = self._image_captions.setdefault(image_id, [])
                    bucket.append(record)

            sample_ids = list(self._caption_lookup.keys())[:5]
            self._caption_samples = [
                self._caption_metadata[cid]
                for cid in sample_ids
                if cid in self._caption_metadata
            ]

            dim = int(self._image_vectors.shape[1])
            self._backends.clear()
            self._backend_errors.clear()
            for method in _DEFAULT_METHODS:
                config = dict(method["config"])
                metric = str(config.get("metric", "l2")).lower()
                try:
                    backend = self._build_backend(
                        backend=str(method["backend"]),
                        dim=dim,
                        config=config,
                    )
                except Exception as exc:  # pragma: no cover - startup logging
                    identifier = str(method["id"])
                    logger.warning(
                        "Skipping backend %s due to error: %s", identifier, exc
                    )
                    self._backend_errors[identifier] = str(exc)
                    continue

                backend.add_embeddings(self._image_ids, self._image_vectors)
                identifier = str(method["id"])
                label = str(method["label"])
                supports_similarity = metric in {"ip", "cosine"}
                self._backends[identifier] = BackendInfo(
                    identifier=identifier,
                    label=label,
                    backend=str(method["backend"]),
                    metric=metric,
                    config=config,
                    searcher=backend,
                    supports_similarity=supports_similarity,
                )
                logger.info("Built %s backend (%s)", identifier, label)

            if not self._backends:
                raise RuntimeError(
                    "No retrieval backends were successfully initialised."
                )

            self._ready = True
            logger.info("Flickr30k demo initialisation complete")

    def _build_backend(
        self, *, backend: str, dim: int, config: Mapping[str, Any]
    ) -> Any:
        backend = backend.lower()
        if backend == "faiss":
            return FaissSearchIndex(dim, config=config)
        if backend == "scann":
            if not scann_available():
                raise RuntimeError(
                    "ScaNN backend requested but the 'scann' package is unavailable"
                )
            return ScannSearchIndex(dim, config=ScannIndexConfig.from_mapping(config))
        if backend == "hnswlib":
            if not hnswlib_available():
                raise RuntimeError(
                    "HNSWlib backend requested but the 'hnswlib' package is unavailable"
                )
            return HnswlibSearchIndex(
                dim, config=HnswlibIndexConfig.from_mapping(config)
            )
        raise ValueError(f"Unsupported backend: {backend}")

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.prepare)

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Search engine is not ready yet")

    def backend_info(self, identifier: str) -> BackendInfo:
        self._require_ready()
        backend = self._backends.get(identifier)
        if backend is None:
            raise KeyError(identifier)
        return backend

    def image_embedding(self, image_id: str) -> np.ndarray:
        self._require_ready()
        index = self._image_lookup.get(image_id)
        if index is None:
            raise KeyError(image_id)
        assert self._image_vectors is not None
        return np.asarray(self._image_vectors[index], dtype="float32")

    def caption_embedding(
        self, caption_id: str
    ) -> tuple[np.ndarray, CaptionRecord | None]:
        self._require_ready()
        index = self._caption_lookup.get(caption_id)
        if index is None:
            raise KeyError(caption_id)
        assert self._caption_vectors is not None
        vector = np.asarray(self._caption_vectors[index], dtype="float32")
        record = self._caption_metadata.get(caption_id)
        return vector, record

    def image_captions(self, image_id: str) -> list[CaptionRecord]:
        self._require_ready()
        return list(self._image_captions.get(image_id, []))

    def caption_detail(self, caption_id: str) -> CaptionRecord | None:
        self._require_ready()
        return self._caption_metadata.get(caption_id)

    def encode_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("Text query must not be empty")
        encoder = self._encoder
        if encoder is None:
            with self._encoder_lock:
                encoder = self._encoder
                if encoder is None:
                    encoder = JinaV4Encoder()
                    self._encoder = encoder
        assert encoder is not None
        with torch.inference_mode():
            encoded = encoder.batch_encode(texts=[text])
        if isinstance(encoded, list):
            tensors: list[torch.Tensor] = []
            for item in encoded:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.detach())
                else:
                    tensors.append(torch.as_tensor(item))
            tensor = torch.stack(tensors)
        elif isinstance(encoded, torch.Tensor):
            tensor = encoded.detach()
        else:
            tensor = torch.as_tensor(encoded)
        array = tensor.cpu().numpy().astype("float32")
        if array.ndim == 2 and array.shape[0] == 1:
            return array[0]
        if array.ndim == 1:
            return array
        return array.reshape(-1)

    def run_search(
        self, *, backend_id: str, query_vector: np.ndarray, top_k: int
    ) -> tuple[BackendInfo, list[dict[str, Any]]]:
        self._require_ready()
        backend = self.backend_info(backend_id)
        matrix = np.asarray(query_vector, dtype="float32")
        if matrix.ndim == 1:
            matrix = np.expand_dims(matrix, axis=0)
        hits = backend.searcher.search(matrix, top_k=top_k)
        if isinstance(hits, list) and hits and isinstance(hits[0], dict):
            rows = hits
        elif isinstance(hits, list):
            rows = hits[0] if hits else []
        else:
            rows = list(hits)
        results: list[dict[str, Any]] = []
        for rank, item in enumerate(rows, start=1):
            identifier = str(item.get("id"))
            distance = float(item.get("distance", 0.0))
            score = distance if backend.supports_similarity else -distance
            captions = [
                {
                    "id": record.id,
                    "caption": record.caption,
                }
                for record in self._image_captions.get(identifier, [])[:5]
            ]
            image_rel = self.settings.image_pattern.format(id=identifier)
            image_path = self.settings.image_root / image_rel
            image_url: str | None = None
            if self.serve_images and image_path.exists():
                image_url = f"/images/{Path(image_rel).as_posix()}"
            results.append(
                {
                    "rank": rank,
                    "id": identifier,
                    "distance": distance,
                    "score": score,
                    "metric": backend.metric,
                    "caption": captions[0]["caption"] if captions else None,
                    "captions": captions,
                    "image_url": image_url,
                    "image_path": str(image_path),
                }
            )
        return backend, results

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "image_count": len(self._image_ids),
            "caption_count": len(self._caption_lookup),
            "image_root": (
                str(self.settings.image_root)
                if self.settings.image_root
                else None
            ),
            "image_pattern": self.settings.image_pattern,
            "methods": [
                {
                    "id": info.identifier,
                    "label": info.label,
                    "backend": info.backend,
                    "metric": info.metric,
                }
                for info in self._backends.values()
            ],
            "method_errors": dict(self._backend_errors),
            "sample_captions": [
                {
                    "id": record.id,
                    "image_id": record.image_id,
                    "caption": record.caption,
                }
                for record in self._caption_samples
            ],
            "images_served": self.serve_images,
            "default_top_k": self.settings.default_top_k,
            "max_top_k": self.settings.max_top_k,
        }
class CaptionPreview(BaseModel):
    id: str
    caption: str | None = None


class ImageHit(BaseModel):
    rank: int
    id: str
    score: float
    distance: float
    metric: str
    caption: str | None = None
    captions: list[CaptionPreview] = Field(default_factory=list)
    image_url: str | None = None
    image_path: str | None = None


class SearchRequest(BaseModel):
    backend: str = Field(default="faiss_flat")
    top_k: int = Field(default=9, ge=1, le=200)
    query_mode: str = Field(default="text")
    query: str | None = None
    caption_id: str | None = None
    image_id: str | None = None

    model_config = {"extra": "forbid"}


class SearchResponse(BaseModel):
    backend: str
    backend_label: str
    metric: str
    top_k: int
    query: dict[str, Any]
    backend_config: dict[str, Any]
    results: list[ImageHit]


class StatusResponse(BaseModel):
    ready: bool
    image_count: int
    caption_count: int
    image_root: str | None
    image_pattern: str
    methods: list[dict[str, Any]]
    method_errors: dict[str, str]
    sample_captions: list[dict[str, Any]]
    images_served: bool
    default_top_k: int
    max_top_k: int


class CaptionDetail(BaseModel):
    id: str
    image_id: str | None
    caption: str | None


def create_app(settings: DemoSettings | None = None) -> FastAPI:
    settings = settings or DemoSettings()
    engine = Flickr30kSearchEngine(settings)
    app = FastAPI(title="Flickr30k Search Demo", version="0.1.0")

    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    if settings.image_root.exists():
        app.mount("/images", StaticFiles(directory=settings.image_root), name="images")
        engine.serve_images = True
    else:
        engine.serve_images = False
        logger.warning(
            "Image directory %s does not exist; image URLs will be disabled",
            settings.image_root,
        )

    @app.on_event("startup")
    async def _startup() -> None:
        await engine.ensure_ready()

    @app.get("/", response_class=FileResponse)
    async def index() -> str:
        return FileResponse(templates_dir / "index.html")

    @app.get("/api/status", response_model=StatusResponse)
    async def api_status() -> StatusResponse:
        await engine.ensure_ready()
        return StatusResponse(**engine.status())

    @app.get("/api/captions/{caption_id}", response_model=CaptionDetail)
    async def api_caption(caption_id: str) -> CaptionDetail:
        await engine.ensure_ready()
        record = engine.caption_detail(caption_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Caption not found")
        return CaptionDetail(
            id=record.id,
            image_id=record.image_id,
            caption=record.caption,
        )

    @app.get(
        "/api/images/{image_id}/captions", response_model=list[CaptionPreview]
    )
    async def api_image_captions(image_id: str) -> list[CaptionPreview]:
        await engine.ensure_ready()
        records = engine.image_captions(image_id)
        return [CaptionPreview(id=item.id, caption=item.caption) for item in records]

    @app.post("/api/search", response_model=SearchResponse)
    async def api_search(request: SearchRequest) -> SearchResponse:
        await engine.ensure_ready()
        top_k = min(request.top_k, engine.settings.max_top_k)
        query_summary: dict[str, Any]
        if request.query_mode == "text":
            if not request.query or not request.query.strip():
                raise HTTPException(status_code=400, detail="Text query required")
            vector = await asyncio.to_thread(engine.encode_text, request.query)
            query_summary = {"mode": "text", "text": request.query}
        elif request.query_mode == "caption":
            if not request.caption_id:
                raise HTTPException(status_code=400, detail="caption_id is required")
            try:
                vector, record = engine.caption_embedding(request.caption_id)
            except KeyError as exc:  # pragma: no cover - runtime validation
                raise HTTPException(
                    status_code=404, detail="Caption not found"
                ) from exc
            query_summary = {
                "mode": "caption",
                "id": request.caption_id,
                "image_id": record.image_id if record else None,
                "caption": record.caption if record else None,
            }
        elif request.query_mode == "image":
            if not request.image_id:
                raise HTTPException(status_code=400, detail="image_id is required")
            try:
                vector = engine.image_embedding(request.image_id)
            except KeyError as exc:  # pragma: no cover - runtime validation
                raise HTTPException(
                    status_code=404, detail="Image not found"
                ) from exc
            query_summary = {
                "mode": "image",
                "id": request.image_id,
                "captions": [
                    {"id": item.id, "caption": item.caption}
                    for item in engine.image_captions(request.image_id)
                ],
            }
        else:
            raise HTTPException(status_code=400, detail="Unsupported query_mode")

        backend_info, results = engine.run_search(
            backend_id=request.backend, query_vector=vector, top_k=top_k
        )
        hits = [ImageHit(**item) for item in results]
        return SearchResponse(
            backend=backend_info.identifier,
            backend_label=backend_info.label,
            metric=backend_info.metric,
            top_k=top_k,
            query=query_summary,
            backend_config=dict(backend_info.config),
            results=hits,
        )

    return app


app = create_app()
