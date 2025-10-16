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
from fastapi.responses import HTMLResponse
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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Flickr30k · Jina V4 Search Demo</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      line-height: 1.5;
    }
    body {
      margin: 0;
      padding: 0;
      background: #11131a;
      color: #f5f6fb;
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 1.5rem;
    }
    header {
      margin-bottom: 1.5rem;
    }
    h1 {
      font-size: 2rem;
      margin: 0 0 0.5rem 0;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(320px, 360px) 1fr;
      gap: 1.5rem;
    }
    .panel {
      background: rgba(18, 22, 35, 0.85);
      border: 1px solid rgba(120, 130, 160, 0.25);
      border-radius: 16px;
      padding: 1.25rem;
      backdrop-filter: blur(12px);
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.35);
    }
    label {
      display: block;
      font-weight: 600;
      margin-bottom: 0.35rem;
    }
    input[type=\"text\"],
    input[type=\"number\"],
    select,
    textarea {
      width: 100%;
      padding: 0.65rem 0.75rem;
      border-radius: 10px;
      border: 1px solid rgba(140, 148, 178, 0.35);
      background: rgba(18, 20, 29, 0.85);
      color: inherit;
      box-sizing: border-box;
    }
    textarea {
      min-height: 120px;
      resize: vertical;
    }
    button {
      cursor: pointer;
      border: none;
      border-radius: 999px;
      padding: 0.65rem 1.25rem;
      font-weight: 600;
      color: #0b0e17;
      background: linear-gradient(135deg, #40c8ff, #9d7bff);
      box-shadow: 0 10px 24px rgba(64, 200, 255, 0.35);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 16px 32px rgba(64, 200, 255, 0.45);
    }
    fieldset {
      border: none;
      padding: 0;
      margin: 0 0 1rem 0;
    }
    fieldset legend {
      font-weight: 600;
      margin-bottom: 0.5rem;
    }
    .radio-row {
      display: flex;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .radio-row label {
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      cursor: pointer;
    }
    .actions {
      display: flex;
      gap: 0.75rem;
      align-items: center;
    }
    .status {
      font-size: 0.9rem;
      line-height: 1.4;
      background: rgba(14, 16, 24, 0.6);
      border-radius: 12px;
      padding: 0.75rem 1rem;
      margin-top: 1rem;
    }
    .status strong {
      font-weight: 600;
      color: #8dd7ff;
    }
    #results {
      display: grid;
      gap: 1.25rem;
    }
    .result-card {
      display: grid;
      grid-template-columns: 200px 1fr;
      gap: 1rem;
      background: rgba(14, 18, 28, 0.9);
      border: 1px solid rgba(120, 132, 160, 0.2);
      border-radius: 16px;
      overflow: hidden;
    }
    .result-card img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: cover;
      background: rgba(20, 24, 38, 0.8);
    }
    .result-meta {
      padding: 1rem;
    }
    .result-meta h3 {
      margin: 0 0 0.4rem 0;
      font-size: 1.05rem;
    }
    .result-meta .metric {
      font-size: 0.85rem;
      opacity: 0.7;
    }
    .caption-list {
      margin: 0.75rem 0 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 0.35rem;
    }
    .caption-list li {
      background: rgba(22, 26, 38, 0.65);
      border-radius: 10px;
      padding: 0.5rem 0.65rem;
      font-size: 0.9rem;
    }
    .error {
      color: #ff8aa6;
      margin-top: 0.5rem;
    }
    @media (max-width: 960px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .result-card {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Flickr30k × Jina V4 Retrieval Demo</h1>
      <p>Interactively explore text & image retrieval over the Flickr30k embeddings with multiple ANN backends.</p>
    </header>
    <div class=\"layout\">
      <section class=\"panel\">
        <form id=\"searchForm\">
          <label for=\"backendSelect\">Search backend</label>
          <select id=\"backendSelect\"></select>

          <label for=\"topK\">Top-K results</label>
          <input type=\"number\" id=\"topK\" min=\"1\" max=\"50\" value=\"9\" />

          <fieldset>
            <legend>Query source</legend>
            <div class=\"radio-row\">
              <label><input type=\"radio\" name=\"queryMode\" value=\"text\" checked />Text</label>
              <label><input type=\"radio\" name=\"queryMode\" value=\"caption\" />Caption ID</label>
              <label><input type=\"radio\" name=\"queryMode\" value=\"image\" />Image ID</label>
            </div>
          </fieldset>

          <div id=\"textQueryGroup\">
            <label for=\"textQuery\">Free-form text</label>
            <textarea id=\"textQuery\" placeholder=\"Describe the scene you are looking for...\"></textarea>
          </div>

          <div id=\"captionQueryGroup\" style=\"display:none\;\">
            <label for=\"captionId\">Caption embedding ID</label>
            <div class=\"actions\">
              <input type=\"text\" id=\"captionId\" placeholder=\"e.g. 12345\" />
              <button type=\"button\" id=\"previewCaption\">Preview</button>
            </div>
            <div id=\"captionPreview\" class=\"status\" style=\"display:none\;\"></div>
          </div>

          <div id=\"imageQueryGroup\" style=\"display:none\;\">
            <label for=\"imageId\">Image ID</label>
            <input type=\"text\" id=\"imageId\" placeholder=\"e.g. 1000092795\" />
          </div>

          <div class=\"actions\" style=\"margin-top:1.25rem\;\">
            <button type=\"submit\">Run search</button>
            <span id=\"formError\" class=\"error\"></span>
          </div>
        </form>

        <div class=\"status\" id=\"statusInfo\"></div>
      </section>

      <section class=\"panel\">
        <h2>Results</h2>
        <div id=\"results\"></div>
      </section>
    </div>
  </main>
  <script>
    const backendSelect = document.getElementById('backendSelect');
    const topKInput = document.getElementById('topK');
    const searchForm = document.getElementById('searchForm');
    const textQueryGroup = document.getElementById('textQueryGroup');
    const captionQueryGroup = document.getElementById('captionQueryGroup');
    const imageQueryGroup = document.getElementById('imageQueryGroup');
    const textQueryInput = document.getElementById('textQuery');
    const captionIdInput = document.getElementById('captionId');
    const imageIdInput = document.getElementById('imageId');
    const statusInfo = document.getElementById('statusInfo');
    const resultsContainer = document.getElementById('results');
    const formError = document.getElementById('formError');
    const captionPreview = document.getElementById('captionPreview');

    function escapeHtml(unsafe) {
      if (unsafe === null || unsafe === undefined) return '';
      return String(unsafe)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    async function loadStatus() {
      try {
        const response = await fetch('/api/status');
        if (!response.ok) {
          throw new Error('Unable to load status');
        }
        const data = await response.json();
        renderStatus(data);
        populateBackends(data.methods);
        topKInput.value = data.default_top_k;
        topKInput.max = data.max_top_k;
        window.flickrDemoStatus = data;
      } catch (err) {
        statusInfo.textContent = err.message;
      }
    }

    function populateBackends(methods) {
      backendSelect.innerHTML = '';
      methods.forEach((method, index) => {
        const option = document.createElement('option');
        option.value = method.id;
        option.textContent = `${method.label} · ${method.metric}`;
        if (index === 0) {
          option.selected = true;
        }
        backendSelect.appendChild(option);
      });
    }

    function renderStatus(data) {
      const served = data.images_served ? 'served locally' : 'not served (files missing)';
      let html = `<div><strong>Embeddings:</strong> ${data.image_count.toLocaleString()} images · ${data.caption_count.toLocaleString()} captions</div>`;
      html += `<div><strong>Images:</strong> ${served}</div>`;
      if (data.sample_captions && data.sample_captions.length) {
        html += '<div style="margin-top:0.75rem"><strong>Sample caption IDs:</strong><ul style="margin:0.35rem 0 0 1.1rem; padding:0; list-style:disc;">';
        data.sample_captions.forEach((item) => {
          html += `<li><code>${escapeHtml(item.id)}</code> → <code>${escapeHtml(item.image_id ?? 'N/A')}</code> — ${escapeHtml(item.caption ?? '(no text)')}</li>`;
        });
        html += '</ul></div>';
      }
      if (Object.keys(data.method_errors || {}).length) {
        html += '<div class="error" style="margin-top:0.75rem">Unavailable backends:<ul style="margin:0.35rem 0 0 1.1rem; padding:0; list-style:disc;">';
        for (const [key, value] of Object.entries(data.method_errors)) {
          html += `<li><code>${escapeHtml(key)}</code> – ${escapeHtml(value)}</li>`;
        }
        html += '</ul></div>';
      }
      statusInfo.innerHTML = html;
    }

    function updateQueryMode() {
      const mode = document.querySelector('input[name="queryMode"]:checked').value;
      textQueryGroup.style.display = mode === 'text' ? 'block' : 'none';
      captionQueryGroup.style.display = mode === 'caption' ? 'block' : 'none';
      imageQueryGroup.style.display = mode === 'image' ? 'block' : 'none';
      formError.textContent = '';
    }

    async function previewCaption() {
      const id = captionIdInput.value.trim();
      captionPreview.style.display = 'none';
      captionPreview.textContent = '';
      if (!id) {
        return;
      }
      try {
        const response = await fetch(`/api/captions/${encodeURIComponent(id)}`);
        if (!response.ok) {
          throw new Error('Caption not found');
        }
        const data = await response.json();
        captionPreview.style.display = 'block';
        captionPreview.innerHTML = `<strong>${escapeHtml(data.id)}</strong> → ${escapeHtml(data.image_id ?? 'N/A')}<br/>${escapeHtml(data.caption ?? '(no caption text)')}`;
      } catch (err) {
        captionPreview.style.display = 'block';
        captionPreview.textContent = err.message;
      }
    }

    async function runSearch(event) {
      event.preventDefault();
      formError.textContent = '';
      const payload = {
        backend: backendSelect.value,
        top_k: Number(topKInput.value) || 9,
        query_mode: document.querySelector('input[name="queryMode"]:checked').value,
      };
      if (payload.query_mode === 'text') {
        payload.query = textQueryInput.value.trim();
        if (!payload.query) {
          formError.textContent = '请输入搜索文本';
          return;
        }
      } else if (payload.query_mode === 'caption') {
        payload.caption_id = captionIdInput.value.trim();
        if (!payload.caption_id) {
          formError.textContent = '请输入 caption ID';
          return;
        }
      } else {
        payload.image_id = imageIdInput.value.trim();
        if (!payload.image_id) {
          formError.textContent = '请输入 image ID';
          return;
        }
      }

      try {
        const response = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          const message = await response.text();
          throw new Error(message || '搜索失败');
        }
        const data = await response.json();
        renderResults(data);
      } catch (err) {
        formError.textContent = err.message;
      }
    }

    function renderResults(data) {
      resultsContainer.innerHTML = '';
      if (!data.results.length) {
        const empty = document.createElement('p');
        empty.textContent = '未检索到结果。';
        resultsContainer.appendChild(empty);
        return;
      }
      data.results.forEach((hit) => {
        const card = document.createElement('article');
        card.className = 'result-card';

        if (hit.image_url) {
          const img = document.createElement('img');
          img.src = hit.image_url;
          img.alt = escapeHtml(hit.caption || hit.id);
          card.appendChild(img);
        } else {
          const placeholder = document.createElement('div');
          placeholder.style.display = 'flex';
          placeholder.style.alignItems = 'center';
          placeholder.style.justifyContent = 'center';
          placeholder.style.background = 'rgba(26,30,44,0.75)';
          placeholder.textContent = 'Image unavailable';
          card.appendChild(placeholder);
        }

        const meta = document.createElement('div');
        meta.className = 'result-meta';
        meta.innerHTML = `<h3>#${hit.rank} · ${escapeHtml(hit.id)}</h3>` +
          `<div class="metric">metric: ${escapeHtml(hit.metric)} · score: ${hit.score.toFixed(4)} · raw: ${hit.distance.toFixed(4)}</div>` +
          (hit.caption ? `<p style="margin:0.6rem 0 0 0;">${escapeHtml(hit.caption)}</p>` : '');

        if (hit.captions && hit.captions.length) {
          const list = document.createElement('ul');
          list.className = 'caption-list';
          hit.captions.forEach((entry) => {
            const item = document.createElement('li');
            item.innerHTML = `<code>${escapeHtml(entry.id)}</code> — ${escapeHtml(entry.caption ?? '(no text)')}`;
            list.appendChild(item);
          });
          meta.appendChild(list);
        }

        if (hit.image_path) {
          const pathInfo = document.createElement('div');
          pathInfo.style.marginTop = '0.65rem';
          pathInfo.style.fontSize = '0.8rem';
          pathInfo.style.opacity = '0.65';
          pathInfo.textContent = hit.image_path;
          meta.appendChild(pathInfo);
        }

        card.appendChild(meta);
        resultsContainer.appendChild(card);
      });
    }

    document.querySelectorAll('input[name="queryMode"]').forEach((input) => {
      input.addEventListener('change', updateQueryMode);
    });
    document.getElementById('previewCaption').addEventListener('click', previewCaption);
    searchForm.addEventListener('submit', runSearch);
    loadStatus();
  </script>
</body>
</html>
"""  # noqa: E501


def create_app(settings: DemoSettings | None = None) -> FastAPI:
    settings = settings or DemoSettings()
    engine = Flickr30kSearchEngine(settings)
    app = FastAPI(title="Flickr30k Search Demo", version="0.1.0")

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

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML_TEMPLATE

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
