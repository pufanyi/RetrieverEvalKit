# System Architecture Overview

This reference explains how RetrieverEvalKit is structured and which supporting
libraries it depends on so you can navigate the repository quickly.

## Environment and tooling

RetrieverEvalKit is packaged as a Python module with dependencies managed by
[uv](https://github.com/astral-sh/uv). Sync everything—including optional ANN
libraries, Streamlit, pytest, and Ruff—by running:

```bash
uv sync --dev
```

The repository targets Python 3.11+ and expects GPU-capable hardware for
multi-process embedding runs. Hydra drives every entrypoint, so add
`+key=value` overrides to adapt the configuration without editing files.

## Directory map

| Path | Purpose |
| --- | --- |
| `img_search/data/` | Dataset abstractions for images, captions, and on-disk embedding corpora.【F:img_search/data/dataset.py†L1-L63】【F:img_search/data/embeddings.py†L1-L120】 |
| `img_search/embedding/` | Encoder implementations (SigLIP, SigLIP2, Jina CLIP, vLLM-backed variants) and the registry helper that instantiates them from Hydra configs.【F:img_search/embedding/__init__.py†L1-L44】 |
| `img_search/pipeline/` | Hydra entrypoints for image and caption embedding jobs with Accelerate-aware retry logic.【F:img_search/pipeline/embed.py†L20-L208】【F:img_search/pipeline/embed_text.py†L20-L227】 |
| `img_search/search/` | ANN benchmarking utilities plus backend-specific wrappers for FAISS, ScaNN, and HNSWlib.【F:img_search/search/evaluate.py†L1-L205】 |
| `img_search/frontend/` | Streamlit application for browsing Flickr30k retrieval results and testing live queries.【F:img_search/frontend/flickr30k_demo.py†L1-L156】 |
| `img_search/config/` | Hydra configuration groups for models, datasets, tasks, logging, and search evaluation presets.【F:img_search/config/embed_config.yaml†L1-L37】【F:img_search/config/models/jina_v4.yaml†L1-L6】 |
| `scripts/` | CLI helpers for benchmarking and publishing embedding artifacts to the Hugging Face Hub.【F:scripts/run_search_eval.py†L1-L60】【F:scripts/upload_to_hub.py†L1-L68】 |

## Core components

Hydra configuration, encoder registries, dataset adapters, and ANN evaluators
cooperate to deliver the full retriever workflow. Use this page alongside the
pipeline and evaluation references when planning new experiments or extending
the toolkit.
