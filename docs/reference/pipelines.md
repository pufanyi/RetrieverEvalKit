# Embedding Pipelines and Dataset Adapters

This reference covers Hydra configuration, pipeline lifecycles, and dataset
interfaces for both image and caption embedding jobs.

## Configuring embedding jobs

The default image pipeline configuration (`img_search/config/embed_config.yaml`)
composes model, dataset, task, logging, and output settings. Override individual
groups at runtime without modifying the base YAML:

```bash
uv run python -m img_search.pipeline.embed \
  models=jina_v4 \
  datasets=inquire \
  output_path=outputs/inquire_jina.parquet
```

Hydra expands `models=jina_v4` into the encoder definition located in
`img_search/config/models/jina_v4.yaml`, which instructs `get_encoder` to
construct a `JinaV4Encoder` with optional Accelerate support.
Similarly, `datasets=inquire` resolves to `img_search/config/datasets/inquire.yaml`,
which feeds parameters to the `InquireDataset`
loader.

### Image pipeline lifecycle

`img_search.pipeline.embed` orchestrates the embedding workflow:

1. Instantiate encoders and datasets from the Hydra config.
2. Use Accelerate to coordinate downloads and retries across all workers so
   models and datasets are built once and cached for subsequent ranks.
3. Iterate over dataset batches, gather embeddings from every process, and write
   them to a Parquet file with identifier, model name, dataset name, and vector
   columns.

Batches are sized via `tasks.batch_size`, and the writer is initialised lazily so
multiple models can append to the same Parquet output when required.

### Caption pipeline lifecycle

`img_search.pipeline.embed_text` mirrors the image flow for caption datasets:

1. Load encoders and a text dataset (e.g., `CaptionsJsonlDataset`).
2. Build the dataset on the main process, broadcast readiness to workers, and
   batch captions into tensors for encoding.
3. Gather embeddings, recover the full caption metadata via `CaptionRecord`, and
   append to a Parquet table containing caption/image identifiers and raw
   text.

Hydra configs for text datasets live under `img_search/config/text_dataset/` and
can be combined with any encoder spec registered in `img_search/embedding/__init__.py`.

### Dataset adapters

- **INQUIRE** – Streams the `evendrow/INQUIRE-Rerank` dataset from the Hugging
  Face Hub, resolves stable identifiers, and exposes batched PIL
  images.
- **Flickr30k captions** – Reads newline-delimited JSON (`caption`, `image`
  fields) and converts each string into a `CaptionRecord` with consistent IDs for
  downstream joins.

Implementing a new dataset involves subclassing `ImageDataset` or `TextDataset`,
defining `build`, `length`, and iterator methods, and registering the class in
`DATASETS` or `TEXT_DATASETS` within `img_search/data/__init__.py` so Hydra configs can reference it.
