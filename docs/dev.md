# Development

## Install Dependencies

```bash
uv sync --dev
```

## Run Pre-commit

Before committing, run the following command to check the code style:

```bash
uv run pre-commit run --all-files
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
