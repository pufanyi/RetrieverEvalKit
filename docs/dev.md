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

## Data Parallel Embedding

Encoders opt into multi-GPU execution by default (`data_parallel=true`) whenever more than
one CUDA device is visible. Pick the devices you want to use by exporting
`CUDA_VISIBLE_DEVICES` before launching the pipeline:

```bash
export CUDA_VISIBLE_DEVICES=0,1  # choose the GPUs that should participate
uv run python -m img_search.pipeline.embed \
  models=siglip2
```

The same applies to `jina_v4` and future DP-aware encoders:

```bash
uv run python -m img_search.pipeline.embed \
  models=jina_v4
```

The default Hydra config resolves to the `jinaai/jina-embeddings-v4-vllm-retrieval`
checkpoint so the pooling runner is available when we call `LLM.encode()`. If you override
`models.0.model_name`, make sure to pick a pooling-capable variant.

Optional tweaks:

- Force a specific primary GPU with `models.0.kwargs.device=cuda:1`.
- Disable DP by overriding `models.0.kwargs.data_parallel=false`.
- Outputs always come back on CPU—move them to another device as needed downstream.
