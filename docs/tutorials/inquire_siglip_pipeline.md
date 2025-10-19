# INQUIRE SigLIP Pipeline

This guide walks through producing SigLIP embeddings for the INQUIRE benchmark and
loading the published Hugging Face dataset (`pufanyi/inquire-siglip-so400m-14-384`)
inside the ANN evaluation tools that ship with this repository.

## Generate SigLIP Vectors

The `InquireDataset` loader now surfaces the `inat24_file_name` column as a stable
identifier so each image is saved with a deterministic ID. Run the embedding pipeline
with the SigLIP encoder to produce gallery vectors:

```bash
uv run python -m img_search.pipeline.embed \
  models=siglip \
  datasets=inquire \
  tasks.batch_size=256 \
  output_path=outputs/inquire/siglip-gallery.parquet
```

- Override `tasks.batch_size` to match available memory.
- Use `models.0.kwargs.device=cpu` if GPUs are unavailable.
- The output Parquet file contains `id`, `model_name`, `dataset_name`, and `embedding`
  columns that mirror the Flickr30k pipeline.

## Convert Shards for the Hub

When working with the raw INQUIRE release (`data/inquire/data/{metadata,img_emb}`), use
the conversion helper to materialise Hugging Face friendly Parquet shards. Remember to
override the model metadata so consumers can distinguish SigLIP vectors from Jina
embeddings:

```bash
uv run python scripts/prepare_data/convert_inquire_to_hf.py \
  --metadata-dir data/inquire/data/metadata \
  --embeddings-dir data/inquire/data/img_emb \
  --output-dir outputs/inquire/hf \
  --model-name google/siglip-so400m-patch14-384 \
  --dataset-name inquire_siglip
```

Pass `--push-to-hub` along with the `--hf-*` arguments to upload shards directly to the
Hub. Gallery embeddings published at
`pufanyi/inquire-siglip-so400m-14-384` follow this layout.

Query vectors for the benchmark live in the same Hub dataset under the `queries`
configuration. They store SigLIP text embeddings alongside the INQUIRE metadata with an
`image_id` relevance column referencing the gallery.

## Generate Query Embeddings

SigLIP text vectors for INQUIRE queries can be materialised directly from the public
CSV releases. The helper script downloads the annotations, encodes the query texts with
SigLIP, and emits Hugging Face compatible Parquet shards:

```bash
uv run python scripts/prepare_data/embed_inquire_queries.py \
  --output-dir outputs/inquire/query_embeddings \
  --dataset-name inquire_siglip \
  --batch-size 128 \
  --chunk-size 1024 \
  --overwrite
```

- `--splits` defaults to `test` and `val`; pass a subset to restrict processing.
- Set `--device cpu` when GPUs are unavailable, or override `--model-name` for
  experimentation.
- Use `--limit` to run quick smoke tests without processing the full benchmark.

Push the generated shards to the Hub with:

```bash
uv run python scripts/prepare_data/embed_inquire_queries.py \
  --push-to-hub \
  --hf-repo-id pufanyi/inquire-siglip-so400m-queries \
  --hf-token $HUGGINGFACE_TOKEN \
  --overwrite
```

The upload registers a `queries` configuration containing `image_id` relevance lists that
match the gallery identifiers, so the search benchmark can join the splits without
additional preprocessing.

## Evaluate Retrieval

Hydra presets are bundled for the new dataset. Point the search benchmark at the SigLIP
configs to reproduce the FAISS/ScaNN numbers:

```bash
uv run python -m img_search.search.evaluate \
  image_dataset=inquire_siglip \
  query_dataset=inquire_siglip \
  evaluation=faiss_only
```

- The default evaluation preset (`evaluation=default`) compares FAISS (Flat/IVF/HNSW),
  ScaNN, and HNSWlib side by side; switch to `faiss_only` for quicker sweeps.
- Override `evaluation.recall_at`, `evaluation.top_k`, or individual method settings
  inline; for example add
  `+evaluation.methods.1='{backend: faiss, method: ivf_pq, metric: cosine, nlist: 2048, nprobe: 32}'`
  to explore IVF-PQ.
- Both dataset configs default to the hosted Hub repo. Swap `image_dataset` or
  `query_dataset` to `inquire_local` (or your custom files) when benchmarking local
  shards.
- Persist raw scores with `evaluation.output_path=outputs/inquire/results.csv` to capture
  recall/latency tables for downstream analysis. The CLI also prints a Rich table with
  per-backend metrics.

## Hydra presets and config tweaks

The SigLIP corpora ship with dedicated Hydra descriptors located under
`img_search/config/search_eval`. The `image_dataset/inquire_siglip.yaml` and
`query_dataset/inquire_siglip.yaml` files target the hosted Hub dataset and define the
identifier, embedding, and relevance columns consumed by the ANN harness.【F:img_search/config/search_eval/image_dataset/inquire_siglip.yaml†L1-L10】【F:img_search/config/search_eval/query_dataset/inquire_siglip.yaml†L1-L10】

When running local experiments, clone those YAML files and adjust `load_from_disk` or
`data_files` to point at your Parquet shards. Because the configs map directly onto the
`EmbeddingDatasetSpec` dataclass, you can also enable memmapped extraction with
`memmap_path=outputs/inquire/memmap` to keep RAM usage bounded during evaluation runs.【F:img_search/data/embeddings.py†L15-L198】

The query config defaults to the `test` split. If you want to evaluate validation queries
instead, pass `query_dataset.split=val` on the command line—the CLI reuses the same
configuration structure without any code changes.【F:img_search/search/evaluate.py†L90-L191】

## Spot-checking the dataset

Before pushing embeddings to the Hub, load them back through
`EmbeddingDatasetSpec` (or `datasets.load_dataset("parquet")`) to verify metadata fields and
vector dimensions. The `scripts/run_search_eval.py` command is a convenient smoke test:
it will parse the Hydra configs, materialise vectors, and run at least one FAISS
configuration—failing fast if IDs or embeddings are missing.【F:scripts/run_search_eval.py†L1-L7】【F:img_search/search/evaluate.py†L123-L309】 Use
`evaluation.methods=[]` to perform only the brute-force baseline while you iterate on the
dataset layout.
