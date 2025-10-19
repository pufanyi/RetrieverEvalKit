# ANN Evaluation Playbook

This repository ships a lightweight harness for comparing approximate nearest
neighbour (ANN) search strategies—including FAISS, ScaNN, and HNSWlib—against a
shared set of image and query embeddings. The workflow is configurable via
[Hydra](https://hydra.cc/) YAML files so you can point the benchmark at any
Hugging Face dataset containing pre-computed embeddings.

Optional dependencies are required for ScaNN (`pip install scann`) and HNSWlib
(`pip install hnswlib`). The harness automatically skips methods whose
libraries are not installed and logs a warning so you can run FAISS-only
benchmarks without additional packages.

## Dataset Format

Both the image and query corpora must be available as Hugging Face datasets.
Each dataset should expose one column containing embedding vectors (a list of
floats) and one column identifying each row. Query datasets may optionally
include a relevance column that lists one or more matching image identifiers so
recall and accuracy can be measured automatically.

| Column           | Required | Description                                    |
|------------------|----------|------------------------------------------------|
| `image_id`       | Yes      | Unique identifier or image path.               |
| `embedding`      | Yes      | Embedding vector stored as a list of floats.   |
| `relevant_ids`   | Optional | String or list of matching image identifiers.  |

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
- **`evaluation/*.yaml`** – selects which ANN methods to benchmark and whether
  to force GPU execution for FAISS.

Large corpora can overwhelm RAM when converted into dense NumPy arrays. Both the
image and query dataset specs accept an optional `memmap_path` field; when set,
`extract_embeddings` will materialise vectors into a NumPy `memmap` on disk
instead of keeping them fully in memory. This allows benchmarks with millions of
queries to run using bounded memory at the cost of additional disk I/O. Point
`memmap_path` at a directory with sufficient free space and reuse the same path
across runs to avoid repeatedly allocating new files.

## Running the CLI

The Hydra entry point at `img_search/search/evaluate.py:app` drives the benchmark and is
exposed via both `python -m img_search.search.evaluate` and
`scripts/run_search_eval.py`. Example invocation:

```bash
uv run python -m img_search.search.evaluate \
  image_dataset=flickr30k_jina \
  query_dataset=flickr30k_jina \
  evaluation=default
```

Hydra resolves each group to the YAML file in `img_search/config/search_eval/` and prints a
Rich table summarising backend, latency, accuracy, and recall metrics once the run
completes. Set
`evaluation.output_path=outputs/benchmarks/flickr30k.csv` to capture the same rows to CSV.

When working on slow machines, reduce the number of queries with
`query_dataset.read_batch_size=512` or request memmaps as described above. Both flags feed
directly into `extract_embeddings`, so you can confirm the impact by watching the log
statements emitted during dataset loading.

## Tuning search methods

The `evaluation` group describes a list of ANN method dictionaries that map directly onto
the `BenchmarkSettings.methods` dataclass in code. Each item can specify `backend`
(`faiss`, `scann`, or `hnswlib`), `metric`, and backend-specific parameters (e.g.
`nprobe` for FAISS IVF). The helper `_expand_method_configs` normalises these dictionaries
and automatically adds GPU/CPU variants when FAISS GPUs are available.

Some quick experiments:

- Force FAISS GPU evaluation: `evaluation.use_gpu=true` (falls back to CPU if no GPU is
detected).
- Compare IVF-PQ with different `nlist`/`nprobe` settings by injecting ad-hoc overrides via
  `+evaluation.methods.1='{backend: faiss, method: ivf_pq, metric: cosine, nlist: 4096, nprobe: 32}'`.
- Skip optional dependencies: when ScaNN or HNSWlib are missing, the harness logs a warning
  and silently removes those configurations before running the benchmark.

## Interpreting outputs

Every benchmark run yields two artefacts: the Rich summary table and the optional CSV/Excel
files. The summary prints `backend`, `method`, latency statistics, accuracy, and recall
values for each configuration. When
`evaluation.excel_output_path` is set, `_write_excel` writes both the summary sheet and a
`details` tab containing individual query hits so you can audit rankings downstream.

For smaller sanity checks, you can also request the brute-force baseline only by setting
`evaluation.methods=[]`. The harness still computes accuracy/recall for the exact search
and includes the rows in the exported files, making it easy to compare approximate runs
against the ground truth later.

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
artifacts. The `evaluation` group now ships with several presets:

- `default` – runs FAISS Flat/IVF/HNSW alongside ScaNN and HNSWlib so you can
  compare every backend in one sweep.
- `faiss_only` – restricts the run to FAISS methods while still comparing CPU
  and GPU variants when available.
- `scann_only` – configures the ScaNN builder with the default leaf and reorder
  parameters.
- `hnswlib_only` – evaluates the HNSWlib index with typical graph parameters.

Feel free to copy these YAML files or create new ones tailored to your dataset
characteristics (e.g., adjusting `nlist`, `ef_search`, or ScaNN leaf counts).

Looking for a ready-made corpus? The
[`pufanyi/flickr30k-jina-embeddings-v4`](https://huggingface.co/datasets/pufanyi/flickr30k-jina-embeddings-v4)
dataset on the Hugging Face Hub exposes JinaCLIP embeddings for Flickr30k
images and captions. This repository now ships config stubs so you can evaluate
it out of the box:

```bash
uv run python -m img_search.search.evaluate \
  image_dataset=flickr30k_jina \
  query_dataset=flickr30k_jina
```

The image split stores one vector per Flickr30k photo under the `id` column,
while the caption split includes five caption embeddings per image and uses the
`image_id` column for relevance labels.

Need INQUIRE coverage instead? The
[`pufanyi/inquire-siglip`](https://huggingface.co/datasets/pufanyi/inquire-siglip)
dataset mirrors the same layout with SigLIP embeddings for the INQUIRE gallery
and queries. Point the benchmark at the new config pair to load it directly
from the Hub:

```bash
uv run python -m img_search.search.evaluate \
  image_dataset=inquire_siglip \
  query_dataset=inquire_siglip
```

Both configs assume the Hub repo exposes `gallery` and `queries` configurations
with an `image_id` relevance column. Override any field at invocation time if
you maintain a fork with different split names or identifier columns. See
[INQUIRE SigLIP pipeline](../tutorials/inquire_siglip_pipeline.md) for guidance on producing and
publishing SigLIP embeddings for INQUIRE.

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
invocation time. For example, to compare a GPU build of FAISS IVF-PQ with a
custom recall schedule while keeping the ScaNN preset:

```bash
uv run python -m img_search.search.evaluate \
  evaluation.methods='[{backend: faiss, method: ivf_pq, metric: cosine, nlist: 1024, nprobe: 32, m: 8}]' \
  evaluation.use_gpu=true \
  +evaluation.methods.1='{backend: scann, method: scann_default, metric: cosine}' \
  evaluation.recall_at='[1, 5, 10, 20]'
```

The script prints a Rich table summarising timing, accuracy, and recall metrics
per method and optionally saves the raw rows as CSV for downstream analysis. A
brute-force baseline rounds out the table by computing exact similarity scores
with NumPy. The baseline is reported once per metric used in the ANN configs so
you can see the recall ceiling alongside approximate index results.
