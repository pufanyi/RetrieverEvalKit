# RetrieverEvalKit Documentation

This documentation set maps RetrieverEvalKit's components, outlines the
recommended workflows, and captures deep dives for search evaluation and demo
experiences. Use the structure below to jump directly to the material you need
while keeping the README focused on high-level positioning.

## Quick start

Follow the quick-start path when you first clone the repository:

1. **Install dependencies** with `uv sync --dev` to mirror the locked runtime and
development toolchain.
2. **Generate embeddings** by running `uv run python -m img_search.pipeline.embed`
and selecting encoders/datasets via Hydra overrides.
3. **Evaluate retrieval** using `uv run python -m img_search.search.evaluate`
against the produced Parquet artifacts.
4. **Explore results** either by launching the Streamlit UI or exporting
benchmarks for sharing.

## Document map

The documentation is organised into reference material, workflow guides, and
hands-on tutorials:

- **Guides** – Process-oriented articles to help you operate the toolkit day to
day.
  - [Development workflow](guides/development_workflow.md)
  - [ANN evaluation playbook](guides/ann_evaluation.md)
- **Tutorials** – Step-by-step walkthroughs for reproducible experiments.
  - [Flickr30k Streamlit demo](tutorials/flickr30k_streamlit.md)
  - [INQUIRE SigLIP pipeline](tutorials/inquire_siglip_pipeline.md)
- **Reference** – Detailed explanations of architecture, pipelines, and data
artifacts.
  - [System architecture overview](reference/architecture.md)
  - [Embedding pipelines and dataset adapters](reference/pipelines.md)
  - [ANN harness and embedding datasets](reference/evaluation.md)
  - [Publishing embeddings to the Hub](reference/publishing.md)

## Workflow cheatsheet

The sections below summarise the most common flows. Each step links back to the
relevant reference pages so you can dig deeper when needed.

### Create gallery embeddings

1. Choose a model/dataset combination using the [embedding pipeline
   reference](reference/pipelines.md#configuring-embedding-jobs).
2. Run `uv run python -m img_search.pipeline.embed` with Hydra overrides to
   tailor batching, output locations, and optional Accelerate integration.
3. Monitor disk output under `outputs/` for Parquet files that downstream
   components expect.

### Prepare caption or query corpora

1. Review the [text embedding guidance](reference/pipelines.md#caption-pipeline-lifecycle)
   to understand required dataset fields.
2. Launch `uv run python -m img_search.pipeline.embed_text` and point to your
   caption JSONL or dataset config.
3. Verify the generated Parquet tables include identifier and text columns before
   moving to the ANN stage.

### Benchmark ANN methods

1. Define gallery and query specs as outlined in the [ANN harness
   reference](reference/evaluation.md#working-with-embedding-datasets).
2. Execute `uv run python -m img_search.search.evaluate` or the wrapper script in
   `scripts/run_search_eval.py`.
3. Export metrics to CSV/Excel and compare throughput across FAISS, ScaNN, and
   HNSWlib implementations.

### Publish and share results

1. Gather the Parquet outputs from image and caption pipelines.
2. Follow the [publishing guide](reference/publishing.md) to push artifacts to a
   Hugging Face dataset repository.
3. Share Hydra override snippets alongside released embeddings to help others
   reproduce your configuration.