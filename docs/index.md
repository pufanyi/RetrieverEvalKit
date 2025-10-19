# Project Overview

This guide connects the major components exposed by RetrieverEvalKit so you can move
from raw datasets to evaluated retrieval systems and interactive demos.

## Environment and Tooling

RetrieverEvalKit is packaged as a Python module with dependencies managed by
[uv](https://github.com/astral-sh/uv). Sync everything—including optional ANN libraries,
Streamlit, pytest, and Ruff—by running:

```bash
uv sync --dev
```

The repository targets Python 3.11+ and expects GPU-capable hardware for multi-process
embedding runs. Hydra drives every entrypoint, so add `+key=value` overrides to adapt the
configuration without editing files.

## Directory Map

| Path | Purpose |
| --- | --- |
| `img_search/data/` | Dataset abstractions for images, captions, and on-disk embedding corpora.【F:img_search/data/dataset.py†L1-L63】【F:img_search/data/embeddings.py†L1-L120】 |
| `img_search/embedding/` | Encoder implementations (SigLIP, SigLIP2, Jina CLIP, vLLM-backed variants) and the registry helper that instantiates them from Hydra configs.【F:img_search/embedding/__init__.py†L1-L44】 |
| `img_search/pipeline/` | Hydra entrypoints for image and caption embedding jobs with Accelerate-aware retry logic.【F:img_search/pipeline/embed.py†L20-L208】【F:img_search/pipeline/embed_text.py†L20-L227】 |
| `img_search/search/` | ANN benchmarking utilities plus backend-specific wrappers for FAISS, ScaNN, and HNSWlib.【F:img_search/search/evaluate.py†L1-L205】 |
| `img_search/frontend/` | Streamlit application for browsing Flickr30k retrieval results and testing live queries.【F:img_search/frontend/flickr30k_demo.py†L1-L156】 |
| `img_search/config/` | Hydra configuration groups for models, datasets, tasks, logging, and search evaluation presets.【F:img_search/config/embed_config.yaml†L1-L37】【F:img_search/config/models/jina_v4.yaml†L1-L6】 |
| `scripts/` | CLI helpers for benchmarking and publishing embedding artifacts to the Hugging Face Hub.【F:scripts/run_search_eval.py†L1-L60】【F:scripts/upload_to_hub.py†L1-L68】 |

## Configuring Embedding Jobs

The default image pipeline configuration (`img_search/config/embed_config.yaml`) composes
model, dataset, task, logging, and output settings. Override individual groups at runtime
without modifying the base YAML:

```bash
uv run python -m img_search.pipeline.embed \
  models=jina_v4 \
  datasets=inquire \
  output_path=outputs/inquire_jina.parquet
```

Hydra expands `models=jina_v4` into the encoder definition located in
`img_search/config/models/jina_v4.yaml`, which instructs `get_encoder` to construct a
`JinaV4Encoder` with optional Accelerate support.【F:img_search/embedding/__init__.py†L27-L44】
Similarly, `datasets=inquire` resolves to `img_search/config/datasets/inquire.yaml`, which
feeds parameters to the `InquireDataset` loader.【F:img_search/data/__init__.py†L17-L38】【F:img_search/data/inquire.py†L10-L97】

### Image Pipeline Lifecycle

`img_search.pipeline.embed` orchestrates the embedding workflow:

1. Instantiate encoders and datasets from the Hydra config.【F:img_search/pipeline/embed.py†L103-L132】
2. Use Accelerate to coordinate downloads and retries across all workers so models and
   datasets are built once and cached for subsequent ranks.【F:img_search/pipeline/embed.py†L20-L102】
3. Iterate over dataset batches, gather embeddings from every process, and write them to a
   Parquet file with identifier, model name, dataset name, and vector columns.【F:img_search/pipeline/embed.py†L133-L208】

Batches are sized via `tasks.batch_size`, and the writer is initialised lazily so multiple
models can append to the same Parquet output when required.

### Caption Pipeline Lifecycle

`img_search.pipeline.embed_text` mirrors the image flow for caption datasets:

1. Load encoders and a text dataset (e.g., `CaptionsJsonlDataset`).【F:img_search/pipeline/embed_text.py†L170-L198】【F:img_search/data/captions.py†L8-L78】
2. Build the dataset on the main process, broadcast readiness to workers, and batch
   captions into tensors for encoding.【F:img_search/pipeline/embed_text.py†L20-L168】
3. Gather embeddings, recover the full caption metadata via `CaptionRecord`, and append to
   a Parquet table containing caption/image identifiers and raw text.【F:img_search/pipeline/embed_text.py†L170-L227】

Hydra configs for text datasets live under `img_search/config/text_dataset/` and can be
combined with any encoder spec registered in `img_search/embedding/__init__.py`.

### Dataset Adapters

- **INQUIRE** – Streams the `evendrow/INQUIRE-Rerank` dataset from the Hugging Face Hub,
  resolves stable identifiers, and exposes batched PIL images.【F:img_search/data/inquire.py†L10-L97】
- **Flickr30k captions** – Reads newline-delimited JSON (`caption`, `image` fields) and
  converts each string into a `CaptionRecord` with consistent IDs for downstream joins.【F:img_search/data/captions.py†L8-L78】

Implementing a new dataset involves subclassing `ImageDataset` or `TextDataset`, defining
`build`, `length`, and iterator methods, and registering the class in `DATASETS` or
`TEXT_DATASETS` within `img_search/data/__init__.py` so Hydra configs can reference it.【F:img_search/data/dataset.py†L1-L63】【F:img_search/data/__init__.py†L17-L48】

## Working with Embedding Datasets

Use `img_search.data.embeddings.EmbeddingDatasetSpec` to describe corpus locations for
both the ANN harness and the Streamlit demo. The helper handles Hub datasets or local
`load_from_disk` directories, supports chunked iteration, and can materialise embeddings to
NumPy memmaps when datasets exceed RAM.【F:img_search/data/embeddings.py†L1-L166】

`extract_embeddings` returns ID lists and `float32` matrices ready for ANN indexing. When
provided with a `memmap_path`, vectors are streamed to disk so even multi-million row
corpora fit within bounded memory.【F:img_search/data/embeddings.py†L59-L166】

## Evaluating ANN Backends

`img_search.search.evaluate` is the primary CLI for measuring retrieval quality:

1. Load gallery and query datasets using `EmbeddingDatasetSpec` or
   `QueryDatasetSpec` definitions.【F:img_search/search/evaluate.py†L90-L120】
2. Expand configured methods (FAISS, ScaNN, HNSWlib) while respecting optional dependency
   availability.【F:img_search/search/evaluate.py†L64-L142】
3. Build indices, compute recall/throughput, and display results in a Rich table while
   optionally exporting CSV/Excel outputs.【F:img_search/search/evaluate.py†L144-L205】

Preset YAML files under `img_search/config/search_eval/` include ready-to-run combinations
for Flickr30k and INQUIRE embeddings. Toggle GPU execution or tweak backend parameters at
runtime with Hydra overrides (`evaluation.use_gpu=true`, `evaluation.methods=[...]`).

## Flickr30k Streamlit Demo

`img_search/frontend/flickr30k_demo.py` wraps the ANN backends behind an interactive UI:

- Loads image and caption embeddings via `EmbeddingDatasetSpec` descriptors and caches the
  resulting vectors.【F:img_search/frontend/flickr30k_demo.py†L1-L156】
- Builds FAISS, ScaNN, and HNSWlib indices (skipping optional dependencies when missing)
  and exposes them through sidebar selectors.【F:img_search/frontend/flickr30k_demo.py†L26-L152】
- Supports text, caption ID, and image ID queries; text queries are embedded on demand via
  `JinaV4Encoder` for quick experimentation.【F:img_search/frontend/flickr30k_demo.py†L18-L152】

Environment variables (`FLICKR30K_IMAGE_ROOT`, `FLICKR30K_IMAGE_PATTERN`,
`FLICKR30K_CAPTION_CONFIG`) tailor the image source and caption split. See
`docs/flickr30k_demo.md` for screenshots and troubleshooting steps.

## Publishing Embeddings

Once image and caption vectors are generated, `scripts/upload_to_hub.py` pushes the
Parquet outputs to a Hugging Face dataset repository. The script creates the target repo if
needed and uploads `images` and `texts` configs in one call.【F:scripts/upload_to_hub.py†L1-L68】

## Further Reading

- [docs/dev.md](dev.md) – Development workflow, lint/test commands, and Accelerate tips.
- [docs/faiss_benchmark.md](faiss_benchmark.md) – Deep dive into the ANN harness.
- [docs/inquire_siglip.md](inquire_siglip.md) – How to reproduce the INQUIRE SigLIP
  embedding pipeline and publish results.
- [docs/flickr30k_demo.md](flickr30k_demo.md) – Streamlit demo usage notes.
