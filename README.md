# RetrieverEvalKit

RetrieverEvalKit bundles everything required to embed large image corpora, generate
caption embeddings, evaluate approximate nearest neighbour (ANN) backends, and ship a
Streamlit demo for Flickr30k-style retrieval experiments. The toolkit is driven by
[Hydra](https://hydra.cc/) configuration files and leans on uv for reproducible
environments, so production runs match the local developer workflow.

## Feature Highlights

- **Hydra-configurable embedding pipelines** – The image and caption pipelines in
  `img_search.pipeline.embed` and `img_search.pipeline.embed_text` coordinate dataset
  downloads, encoder initialisation, and synchronised batch processing across
  multi-GPU launches with [Accelerate](https://huggingface.co/docs/accelerate).【F:img_search/pipeline/embed.py†L20-L146】【F:img_search/pipeline/embed_text.py†L20-L168】
- **Pluggable dataset adapters** – Built-in loaders cover INQUIRE images and Flickr30k
  captions, with extensible base classes for additional image or text datasets.【F:img_search/data/dataset.py†L1-L63】【F:img_search/data/inquire.py†L10-L97】【F:img_search/data/captions.py†L8-L78】
- **Encoder registry** – Switch between SigLIP, SigLIP2, and Jina CLIP families (plus
  vLLM-backed variants) by editing `img_search/config/models/*.yaml` or passing Hydra
  overrides at runtime.【F:img_search/embedding/__init__.py†L1-L44】【F:img_search/config/models/jina_v4.yaml†L1-L6】
- **Search benchmarking harness** – `img_search.search.evaluate` loads embedding
  datasets, spins up FAISS/ScaNN/HNSWlib indices, and reports timing plus recall
  metrics with optional GPU acceleration.【F:img_search/search/evaluate.py†L1-L205】
- **Streamlit front-end** – The Flickr30k demo wraps the ANN backends behind a browser
  UI and serves local thumbnails when available.【F:img_search/frontend/flickr30k_demo.py†L1-L156】
- **Automation scripts** – Helper CLIs prepare datasets (`scripts/prepare_data/`), run
  ANN evaluations, and push embeddings to the Hugging Face Hub.【F:scripts/run_search_eval.py†L1-L60】【F:scripts/upload_to_hub.py†L1-L68】

## Installation

RetrieverEvalKit targets Python 3.11+ and uses [uv](https://github.com/astral-sh/uv) to
lock dependencies. Install uv first, then sync the environment:

```bash
uv sync --dev
```

The command installs runtime dependencies, Streamlit, optional ANN libraries declared in
`pyproject.toml`, and development tooling such as Ruff and pytest.【F:pyproject.toml†L1-L136】

## Quickstarts

### Embed image datasets

Run the embedding pipeline through Hydra. The default configuration emits Parquet files
under `outputs/embeddings/` with one vector per image and metadata columns for the model
and dataset names.

```bash
uv run python -m img_search.pipeline.embed
```

Key Hydra groups include:

- `models` – selects encoders like `siglip`, `siglip2`, `jina_v4`, or their vLLM
  variants.【F:img_search/embedding/__init__.py†L27-L44】
- `datasets` – describes image datasets such as `inquire` or `flickr30k`, including
  split names and identifier columns.【F:img_search/data/__init__.py†L1-L48】
- `tasks` – controls batching behaviour (e.g. `batch_size`).【F:img_search/pipeline/embed.py†L110-L142】

To accelerate multi-GPU jobs, launch via Accelerate and enable the integration for the
Jina encoder:

```bash
uv run accelerate launch --num_processes 8 --mixed_precision bf16 \
  img_search/pipeline/embed.py \
  models=jina_v4 \
  models.0.kwargs.use_accelerate=true \
  tasks.batch_size=512
```

The pipeline coordinates downloads on the main process and synchronises workers before
embedding batches, retrying transient failures automatically.【F:img_search/pipeline/embed.py†L20-L142】

### Embed caption corpora

Use `img_search.pipeline.embed_text` to transform caption JSONL datasets into Parquet
embedding tables. Caption records carry both caption/image identifiers and the raw text
for downstream analysis.【F:img_search/pipeline/embed_text.py†L95-L168】【F:img_search/data/captions.py†L8-L78】

```bash
uv run python -m img_search.pipeline.embed_text \
  text_dataset.path=data/flickr30k/captions.jsonl \
  models=jina_v4
```

### Benchmark ANN indices

The ANN harness compares FAISS, ScaNN, and HNSWlib on a shared query/gallery corpus and
reports throughput plus recall.

```bash
uv run python -m img_search.search.evaluate \
  image_dataset=flickr30k_jina \
  query_dataset=flickr30k_jina
```

Configuration files live under `img_search/config/search_eval/` and describe dataset
locations, ANN parameters, and recall levels.【F:img_search/config/search_eval/eval.yaml†L1-L5】【F:img_search/search/evaluate.py†L64-L142】

### Explore Flickr30k in the browser

Launch the Streamlit demo to test text and caption queries against Flickr30k embeddings.
It automatically loads the default Hugging Face dataset, builds ANN indices, and renders
local thumbnails when `FLICKR30K_IMAGE_ROOT` points to the Flickr30k image directory.【F:img_search/frontend/flickr30k_demo.py†L1-L156】

```bash
uv run streamlit run img_search/frontend/flickr30k_demo.py
```

Set `FLICKR30K_CAPTION_CONFIG`, `FLICKR30K_IMAGE_ROOT`, or `FLICKR30K_IMAGE_PATTERN` to
customise the data sources.【F:img_search/frontend/flickr30k_demo.py†L53-L96】

## Outputs

Both embedding pipelines emit Apache Parquet files containing Arrow tables with stable
identifiers, encoder/dataset metadata, and dense embedding vectors. These files are
compatible with `datasets.load_dataset("parquet")` and the repository’s upload script for
publishing to the Hugging Face Hub.【F:img_search/pipeline/embed.py†L167-L208】【F:img_search/pipeline/embed_text.py†L170-L227】【F:scripts/upload_to_hub.py†L1-L68】

## Project Layout

- `img_search/data/` – Dataset abstractions for images, captions, and embedding specs.
- `img_search/embedding/` – Encoder implementations and registry helpers.
- `img_search/pipeline/` – Hydra entrypoints for image and caption embedding jobs.
- `img_search/search/` – ANN index builders plus the evaluation CLI.
- `img_search/frontend/` – Streamlit demo for Flickr30k retrieval.
- `img_search/utils/` – Logging utilities and shared helpers.
- `docs/` – Extended guides, including ANN benchmarking and Streamlit usage.
- `scripts/` – CLI utilities for dataset preparation, benchmarking, and Hub uploads.
- `tests/` – Pytest suites covering datasets, encoders, and ANN helpers.

## Documentation

Additional guides live under `docs/`:

- [Development workflow](docs/dev.md) – Environment setup, linting, and test commands.
- [INQUIRE SigLIP pipeline](docs/inquire_siglip.md) – Instructions for producing and
  evaluating SigLIP embeddings for INQUIRE.
- [ANN benchmark harness](docs/faiss_benchmark.md) – Detailed walkthrough of the
  evaluation CLI.
- [Flickr30k Streamlit demo](docs/flickr30k_demo.md) – UI controls, environment
  overrides, and troubleshooting tips.
- [Project overview](docs/index.md) – Component breakdown and end-to-end recipes.

## Development

Run Ruff, pytest, and other pre-commit checks before committing changes. The helper guide
in `docs/dev.md` captures the recommended workflow, including multi-GPU Accelerate tips.

```bash
uv run pre-commit run --all-files
uv run pytest -n auto
```

## Publishing embeddings to the Hub

`python scripts/upload_to_hub.py` uploads image and caption Parquet outputs to a Hugging
Face dataset repository, creating the target repo if necessary. Override the default
paths and repo ID with CLI flags as needed.【F:scripts/upload_to_hub.py†L1-L68】

---

RetrieverEvalKit is released under the Apache 2.0 License. Contributions are welcome—see
existing docs for coding standards and testing expectations.【F:LICENSE†L1-L201】
