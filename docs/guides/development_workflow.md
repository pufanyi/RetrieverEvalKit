# Development Workflow

## Install Dependencies

```bash
uv sync --dev
```

## Run Pre-commit

Before committing, run the following command to check the code style:

```bash
uv run pre-commit run --all-files
```

## Run Tests

Execute the automated test suite with pytest. The `-n auto` flag parallelises the run across all logical CPU cores so the smoke scripts finish quickly:

```bash
uv run pytest -n auto
```

## Multi-GPU Embedding with Accelerate

The `JinaV4Encoder` ships with an optional Accelerate integration so you can fan embedding
workloads across however many GPUs you have (e.g., 8×H100). A typical workflow looks like:

1. Configure Accelerate once (accept defaults unless you have a custom setup):
   ```bash
   accelerate config
   ```
2. Launch the embedding pipeline via `accelerate` and enable the integration through Hydra:
   ```bash
   uv run accelerate launch --num_processes 8 --mixed_precision bf16 \
     img_search/pipeline/embed.py \
     models=jina_v4 \
     models.0.kwargs.use_accelerate=true \
     tasks.batch_size=512
   ```
   - Because `accelerate` executes the script with `python`, pass it the actual module file
     (`img_search/pipeline/embed.py`). Using `uv run python -m ...` causes Python to look
     for a file literally named `uv` and fail.
   - `tasks.batch_size` controls how much work each rank processes; scale it until your
     GPUs stay saturated.
   - Pass any custom `Accelerator` parameters with `models.0.kwargs.accelerator_kwargs`,
     e.g. `{"even_batches": true}`.

If you prefer to run without Accelerate, simply omit `use_accelerate=true`; the encoder
falls back to regular single-process execution and still honours `models.0.kwargs.device`.

### Additional Tips

- Use `CUDA_VISIBLE_DEVICES` to restrict which GPUs participate in the run.
- Monitor utilisation with `nvidia-smi dmon` or `accelerate env --report` so you can tune
  batch sizes and data loading.
- Embeddings are returned on CPU; move them to GPU afterwards if downstream consumers
  expect device tensors.

## Targeted testing

The repository ships a sizeable pytest suite split across dedicated subpackages. Run the
tests most relevant to your change instead of the entire matrix when iterating quickly:

- Encoders: `uv run pytest tests/embedding -k jina` validates the Jina integration and
  catches regressions around Accelerate usage.【F:tests/embedding/test_jina.py†L1-L88】
- Dataset loaders: `uv run pytest tests/data` exercises the dataset registries,
  `InquireDataset`, and caption parsing helpers.【F:tests/data/test_dataset.py†L1-L118】【F:tests/data/test_inquire.py†L1-L101】
- Search harness: `uv run pytest tests/search` spins up FAISS/ScaNN/HNSWlib indexes using
  synthetic embeddings to ensure the CLI stays stable.【F:tests/search/test_evaluate.py†L1-L118】

Adopt the full `uv run pytest -n auto` command before opening a PR to verify the parallel
suite passes on a clean environment.

## Working with configs

All entry points are wired through Hydra, so configuration edits belong in
`img_search/config`. The top-level `embed_config.yaml` composes models, datasets, and task
settings that the pipelines consume at runtime.【F:img_search/config/embed_config.yaml†L1-L37】 Use
`uv run python -m img_search.pipeline.embed models=jina_v4 datasets=inquire` to confirm
your YAML changes resolve correctly without needing to commit them.

For logging tweaks, update `img_search/config/logging/info.yaml` and rely on
`setup_logger` to propagate the settings across Accelerate workers.【F:img_search/config/logging/info.yaml†L1-L28】【F:img_search/utils/logging.py†L1-L118】

## Dataset caches and retries

Large dataset downloads happen once on the main Accelerate rank, then subsequent workers
reuse the cached artefacts. The image and text pipelines coordinate this behaviour via
`safe_build_dataset`, which retries transient failures and blocks until the resource is
ready before other processes continue.【F:img_search/pipeline/embed.py†L56-L104】【F:img_search/pipeline/embed_text.py†L99-L157】 Keep
datasets on a fast local disk (or pre-download them) to avoid repeated cache misses during
development.
