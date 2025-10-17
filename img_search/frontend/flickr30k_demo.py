from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from loguru import logger

# Ensure local package imports work when launched via `streamlit run`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

_DEFAULT_METHOD_MAP: dict[str, dict[str, Any]] = {
    str(method["id"]): method for method in _DEFAULT_METHODS
}


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
            caption_config = os.getenv("FLICKR30K_CAPTION_CONFIG")
            if not caption_config or caption_config == "text":
                caption_config = "texts"
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

        self._data_loaded = False
        self.serve_images = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def prepare(
        self,
        *,
        method_filter: Iterable[str] | None = None,
        progress_callback: Callable[[float, str, float], None] | None = None,
    ) -> dict[str, float]:
        start_time = time.perf_counter()
        timings: dict[str, float] = {}

        if method_filter is None:
            requested_ids = {str(method["id"]) for method in _DEFAULT_METHODS}
        else:
            requested_ids = {str(identifier) for identifier in method_filter}
            unknown = [
                identifier
                for identifier in requested_ids
                if identifier not in _DEFAULT_METHOD_MAP
            ]
            if unknown:
                raise KeyError(
                    "Unknown backend identifiers: {}".format(", ".join(sorted(unknown)))
                )

        if method_filter is None:
            ordered_ids = [str(method["id"]) for method in _DEFAULT_METHODS]
        else:
            ordered_ids = [
                str(method["id"])
                for method in _DEFAULT_METHODS
                if str(method["id"]) in requested_ids
            ]

        with self._build_lock:
            pending_backends = [
                identifier
                for identifier in ordered_ids
                if identifier not in self._backends
            ]
            dataset_steps = 3 if not self._data_loaded else 0
            total_steps = dataset_steps + len(pending_backends)

            if total_steps == 0:
                if progress_callback:
                    progress_callback(
                        1.0,
                        "Vector index already available",
                        time.perf_counter() - start_time,
                    )
                timings["total"] = 0.0
                return timings

            completed_steps = 0

            def notify(fraction: float, message: str) -> None:
                if progress_callback:
                    fraction = max(0.0, min(1.0, fraction))
                    progress_callback(
                        fraction,
                        message,
                        time.perf_counter() - start_time,
                    )

            if not self._data_loaded:
                notify(0.0, "Loading Flickr30k image embeddings...")
                step_start = time.perf_counter()
                logger.info(
                    "Loading Flickr30k image embeddings: {}",
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
                    identifier: index
                    for index, identifier in enumerate(self._image_ids)
                }
                timings["load_image_embeddings"] = time.perf_counter() - step_start
                completed_steps += 1
                notify(completed_steps / total_steps, "Image embeddings ready")
                logger.info(
                    "Loaded {} image embeddings (dim={})",
                    len(self._image_ids),
                    self._image_vectors.shape[1],
                )

                notify(
                    completed_steps / total_steps,
                    "Loading Flickr30k caption embeddings...",
                )
                step_start = time.perf_counter()
                logger.info(
                    "Loading Flickr30k caption embeddings: {}",
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
                    str(identifier): index
                    for index, identifier in enumerate(caption_ids)
                }
                timings["load_caption_embeddings"] = time.perf_counter() - step_start
                completed_steps += 1
                notify(
                    completed_steps / total_steps,
                    "Caption embeddings ready",
                )
                logger.info(
                    "Loaded {} caption embeddings (dim={})",
                    len(self._caption_lookup),
                    self._caption_vectors.shape[1],
                )

                notify(
                    completed_steps / total_steps,
                    "Preparing caption metadata...",
                )
                step_start = time.perf_counter()
                self._caption_metadata.clear()
                self._image_captions.clear()
                for row in caption_dataset:
                    raw_id = row.get(self.settings.caption_dataset.id_column)
                    if raw_id is None:
                        continue
                    caption_id = str(raw_id)
                    image_id_value = row.get("image_id")
                    image_id = (
                        str(image_id_value) if image_id_value is not None else None
                    )
                    caption_text_value = row.get("caption")
                    caption_text = (
                        None if caption_text_value is None else str(caption_text_value)
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
                timings["prepare_caption_metadata"] = time.perf_counter() - step_start
                completed_steps += 1
                notify(completed_steps / total_steps, "Caption metadata prepared")
                self._data_loaded = True
                self._ready = True
                logger.info("Flickr30k embeddings loaded")

            if self._image_vectors is None:
                raise RuntimeError("Image embeddings are not loaded")

            dim = int(self._image_vectors.shape[1])
            for method_id in ordered_ids:
                if method_id not in pending_backends:
                    continue
                method = _DEFAULT_METHOD_MAP[method_id]
                label = str(method["label"])
                notify(completed_steps / total_steps, f"Building {label}...")
                step_start = time.perf_counter()
                config = dict(method["config"])
                metric = str(config.get("metric", "l2")).lower()
                try:
                    backend = self._build_backend(
                        backend=str(method["backend"]),
                        dim=dim,
                        config=config,
                    )
                except Exception as exc:  # pragma: no cover - startup logging
                    logger.warning(
                        "Skipping backend {} due to error: {}",
                        method_id,
                        exc,
                    )
                    self._backend_errors[method_id] = str(exc)
                    timings[f"build_{method_id}"] = time.perf_counter() - step_start
                    completed_steps += 1
                    notify(
                        completed_steps / total_steps,
                        f"Failed to initialise {label}",
                    )
                    continue

                backend.add_embeddings(self._image_ids, self._image_vectors)
                supports_similarity = metric in {"ip", "cosine"}
                self._backends[method_id] = BackendInfo(
                    identifier=method_id,
                    label=label,
                    backend=str(method["backend"]),
                    metric=metric,
                    config=config,
                    searcher=backend,
                    supports_similarity=supports_similarity,
                )
                self._backend_errors.pop(method_id, None)
                timings[f"build_{method_id}"] = time.perf_counter() - step_start
                completed_steps += 1
                notify(completed_steps / total_steps, f"{label} ready")
                logger.info("Built {} backend ({})", method_id, label)

            if method_filter is None:
                if not self._backends:
                    raise RuntimeError(
                        "No retrieval backends were successfully initialised."
                    )
            else:
                missing = [mid for mid in ordered_ids if mid not in self._backends]
                if missing:
                    details = ", ".join(
                        f"{mid}: {self._backend_errors.get(mid, 'Unknown error')}"
                        for mid in missing
                    )
                    raise RuntimeError(
                        f"Failed to initialise requested backends: {details}"
                    )

            notify(1.0, "Flickr30k demo initialisation complete")

        timings["total"] = time.perf_counter() - start_time
        return timings

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
                str(self.settings.image_root) if self.settings.image_root else None
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


@st.cache_resource()
def load_engine(settings: DemoSettings | None = None) -> Flickr30kSearchEngine:
    return Flickr30kSearchEngine(settings or DemoSettings())


def _format_caption_option(record: Mapping[str, Any]) -> str:
    caption = record.get("caption") or "(No text)"
    preview = textwrap.shorten(caption, width=36, placeholder="…")
    image_id = record.get("image_id") or "?"
    return f"{record['id']} · Image {image_id} · {preview}"


def _render_sidebar(status: Mapping[str, Any]) -> tuple[int, str]:
    st.sidebar.header("Retrieval Settings")
    if not status.get("ready"):
        st.sidebar.info("Select a method tab and click Load to initialise an index.")
    else:
        loaded_methods = status.get("methods", [])
        if loaded_methods:
            summary = ", ".join(
                method.get("label", method.get("id", "")) for method in loaded_methods
            )
            st.sidebar.caption(f"Loaded backends: {summary}")

    top_k = st.sidebar.slider(
        "Number of results",
        min_value=3,
        max_value=int(status.get("max_top_k", 50)),
        value=int(status.get("default_top_k", 9)),
        step=1,
    )

    query_mode = st.sidebar.radio(
        "Query Mode",
        options=("text", "image", "caption"),
        format_func=lambda mode: {
            "text": "Text",
            "image": "Image",
            "caption": "Caption",
        }[mode],
        horizontal=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"📦 {status.get('image_count', 0):,} images · "
        f"{status.get('caption_count', 0):,} captions"
    )
    if status.get("method_errors"):
        with st.sidebar.expander("Disabled Retrieval Backends"):
            for key, message in status["method_errors"].items():
                st.warning(f"{key}: {message}")

    return top_k, query_mode


def _execute_search(
    engine: Flickr30kSearchEngine,
    *,
    backend_id: str,
    top_k: int,
    query_mode: str,
    query_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], BackendInfo, list[dict[str, Any]]]:
    if query_mode == "text":
        query = (query_inputs.get("query") or "").strip()
        if not query:
            raise ValueError("Please enter a text description")
        vector = engine.encode_text(query)
        summary = {"mode": "text", "text": query}
    elif query_mode == "image":
        image_id = (query_inputs.get("image_id") or "").strip()
        if not image_id:
            raise ValueError("Please enter an image ID")
        vector = engine.image_embedding(image_id)
        summary = {
            "mode": "image",
            "id": image_id,
            "captions": [
                {"id": item.id, "caption": item.caption}
                for item in engine.image_captions(image_id)
            ],
        }
    else:
        caption_id = (query_inputs.get("caption_id") or "").strip()
        if not caption_id:
            raise ValueError("Please select or enter a caption ID")
        vector, record = engine.caption_embedding(caption_id)
        summary = {
            "mode": "caption",
            "id": caption_id,
            "image_id": record.image_id if record else None,
            "caption": record.caption if record else None,
        }

    backend_info, results = engine.run_search(
        backend_id=backend_id, query_vector=vector, top_k=top_k
    )
    return summary, backend_info, results


def _render_results(
    *,
    summary: Mapping[str, Any],
    backend: BackendInfo,
    results: list[dict[str, Any]],
) -> None:
    st.subheader("Search Results")
    with st.expander("Query Information", expanded=True):
        st.write(
            {
                "Mode": {"text": "Text", "image": "Image", "caption": "Caption"}[
                    summary.get("mode", "text")
                ],
                "Backend": backend.label,
                "Similarity Metric": backend.metric,
            }
        )
        if summary.get("mode") == "text":
            st.markdown(f"**Query Text:** {summary.get('text', '')}")
        elif summary.get("mode") == "image":
            st.markdown(f"**Image ID:** {summary.get('id', '')}")
        else:
            caption_text = summary.get("caption") or "(No text)"
            st.markdown(f"**Caption:** {caption_text}")

    if not results:
        st.info("No matching results found.")
        return

    columns = st.columns(3)
    for index, item in enumerate(results):
        column = columns[index % 3]
        with column:
            st.markdown(f"### #{item['rank']} · Image {item['id']}")
            image_path = Path(item.get("image_path", ""))
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            st.caption(
                f"Similarity: {item['score']:.4f} · "
                f"Distance: {item['distance']:.4f} ({item['metric']})"
            )
            caption = item.get("caption") or "No description"
            st.write(caption)
            extra = item.get("captions", [])[1:]
            if extra:
                with st.expander("More Captions"):
                    for cap in extra:
                        st.markdown(f"- {cap.get('caption') or '(No text)'}")


def main() -> None:
    st.set_page_config(
        page_title="Flickr30k Image Search Demo",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 Flickr30k Image Search")
    st.caption("Fast vector retrieval experience with Jina v4 encoder")

    st.session_state.setdefault("backend_timings", {})

    engine = load_engine()
    status = engine.status()

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Image Count", f"{status.get('image_count', 0):,}")
    stats_col2.metric("Caption Count", f"{status.get('caption_count', 0):,}")
    stats_col3.metric("Loaded Backends", f"{len(status.get('methods', []))}")

    top_k, query_mode = _render_sidebar(status)

    method_specs = list(_DEFAULT_METHODS)
    tab_labels = [str(spec["label"]) for spec in method_specs]
    tabs = st.tabs(tab_labels)

    samples = status.get("sample_captions", [])

    for spec, tab in zip(method_specs, tabs, strict=True):
        method_id = str(spec["id"])
        label = str(spec["label"])
        backend_name = str(spec.get("backend", "")).upper()
        metric = str(spec.get("config", {}).get("metric", "l2"))

        with tab:
            st.markdown(f"### {label}")
            st.caption(f"{backend_name} · Metric: {metric}")

            method_ready = any(
                info.get("id") == method_id for info in status.get("methods", [])
            )
            method_error = status.get("method_errors", {}).get(method_id)

            if method_ready:
                st.success("Vector index ready.")
            elif method_error:
                st.warning(f"Last load failed: {method_error}")
            else:
                st.info("Vector index not loaded yet.")

            load_clicked = st.button(
                f"Load {label} index",
                key=f"load_backend_{method_id}",
                use_container_width=True,
            )

            if load_clicked:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(
                    0.0, text="Preparing resources..."
                )

                def on_progress(fraction: float, message: str, elapsed: float) -> None:
                    safe_value = max(0.0, min(1.0, fraction))
                    progress_bar.progress(
                        safe_value, text=f"{message} ({elapsed:.2f}s)"
                    )
                    status_placeholder.caption(f"Elapsed: {elapsed:.2f}s")

                try:
                    with st.spinner("Loading vector index..."):
                        timings = engine.prepare(
                            method_filter=[method_id],
                            progress_callback=on_progress,
                        )
                except Exception as exc:  # pragma: no cover - interactive feedback
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    st.error(f"Failed to load backend: {exc}")
                    logger.exception("Backend initialisation failed")
                    status = engine.status()
                    method_error = status.get("method_errors", {}).get(method_id)
                    method_ready = False
                else:
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    status = engine.status()
                    samples = status.get("sample_captions", [])
                    st.session_state.setdefault("backend_timings", {})[method_id] = (
                        timings
                    )
                    st.session_state["active_backend"] = method_id
                    method_ready = True
                    method_error = None
                    st.success(f"{label} loaded in {timings.get('total', 0.0):.2f}s")

            timings_store = st.session_state.get("backend_timings", {}).get(method_id)
            if timings_store:
                st.caption("Most recent load timings (seconds)")
                timing_rows = {
                    key: f"{value:.2f}s" for key, value in timings_store.items()
                }
                st.write(timing_rows)

            if not method_ready:
                continue

            current_samples = samples
            query_inputs: dict[str, Any] = {}
            with st.form(f"search_form_{method_id}", clear_on_submit=False):
                st.subheader("Enter Query")
                if query_mode == "text":
                    query_inputs["query"] = st.text_area(
                        "Please enter the text description to retrieve",
                        placeholder="A golden retriever running on the grass",
                        key=f"text_query_{method_id}",
                    )
                elif query_mode == "image":
                    query_inputs["image_id"] = st.text_input(
                        "Please enter an image ID",
                        placeholder="Example: 1000092795",
                        key=f"image_id_input_{method_id}",
                    )
                else:
                    selected_id = ""
                    if current_samples:
                        labels = [
                            _format_caption_option(item) for item in current_samples
                        ]
                        selected_label = st.selectbox(
                            "Select sample caption",
                            labels,
                            key=f"caption_sample_{method_id}",
                        )
                        selected_index = labels.index(selected_label)
                        selected_id = current_samples[selected_index]["id"]
                        st.caption("You can also enter a caption ID below to override.")
                    manual_id = st.text_input(
                        "Caption ID",
                        value="",
                        key=f"caption_id_input_{method_id}",
                    )
                    query_inputs["caption_id"] = manual_id.strip() or selected_id

                submitted = st.form_submit_button(
                    "Start Search",
                    use_container_width=True,
                )

            if not submitted:
                continue

            st.session_state["active_backend"] = method_id
            try:
                with st.spinner("Searching..."):
                    summary, backend_info, results = _execute_search(
                        engine,
                        backend_id=method_id,
                        top_k=top_k,
                        query_mode=query_mode,
                        query_inputs=query_inputs,
                    )
            except KeyError as exc:
                st.error(f"Could not find the specified ID: {exc}")
                logger.exception("Missing identifier")
                continue
            except ValueError as exc:
                st.warning(str(exc))
                continue
            except Exception as exc:  # pragma: no cover - interactive feedback
                st.error(f"An error occurred during the search process: {exc}")
                logger.exception("Search failed")
                continue

            _render_results(summary=summary, backend=backend_info, results=results)


if __name__ == "__main__":
    main()
