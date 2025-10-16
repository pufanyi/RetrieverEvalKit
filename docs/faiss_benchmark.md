# FAISS Benchmark Harness

This repository ships a lightweight harness for comparing multiple FAISS search
strategies against a shared set of image and query embeddings. The workflow is
configurable via [Hydra](https://hydra.cc/) YAML files so you can point the
benchmark at any Hugging Face dataset containing pre-computed embeddings.

## Dataset Format

Both the image and query corpora must be available as Hugging Face datasets.
Each dataset should expose one column containing embedding vectors (a list of
floats) and one column identifying each row. Query datasets may optionally
include a relevance column that lists one or more matching image identifiers so
recall and accuracy can be measured automatically.

```
| Column           | Required | Description                                    |
|------------------|----------|------------------------------------------------|
| `image_id`       | Yes      | Unique identifier or image path.               |
| `embedding`      | Yes      | Embedding vector stored as a list of floats.   |
| `relevant_ids`   | Optional | String or list of matching image identifiers.  |
```

You can author the datasets in any way that Hugging Face supports. For local
experimentation, create them with `datasets.Dataset.from_dict` and
`Dataset.save_to_disk`. Remote datasets hosted on the Hub can be referenced
through their repository names instead.

## Configuration Files

The harness pulls settings from three Hydra config groups located under
`img_search/config/search_eval/`:

- **`image_dataset/*.yaml`** – points to the embedding dataset for the gallery.
- **`query_dataset/*.yaml`** – points to the query embeddings and optional
ground-truth column.
- **`evaluation/*.yaml`** – selects which FAISS methods to benchmark and whether
to force GPU execution.

The default composition is described in `img_search/config/search_eval/eval.yaml`:

```yaml
defaults:
  - image_dataset: local_demo
  - query_dataset: local_demo
  - evaluation: default
  - _self_
```

Update the `load_from_disk` path (or switch to a Hub dataset name) in
`image_dataset/local_demo.yaml` and `query_dataset/local_demo.yaml` to match your
artifacts. The evaluation config already lists three FAISS methods (flat,
IVF-Flat, HNSW) and will write the consolidated metrics to
`outputs/faiss_benchmark.csv`.

## Running the Benchmark

Once the datasets and configs are in place, launch the benchmark either through
the module entry point:

```bash
uv run python -m img_search.search.evaluate
```

or with the convenience script:

```bash
uv run scripts/run_search_eval.py
```

Hydra exposes the usual override syntax so you can swap datasets or settings at
invocation time. For example, to compare a GPU build of IVF-PQ with a different
recall schedule:

```bash
uv run python -m img_search.search.evaluate \
  evaluation.methods='[{method: ivf_pq, metric: cosine, nlist: 1024, nprobe: 32, m: 8}]' \
  evaluation.use_gpu=true \
  evaluation.recall_at='[1, 5, 10, 20]'
```

The script prints a Rich table summarising timing, accuracy, and recall metrics
per method and optionally saves the raw rows as CSV for downstream analysis.
