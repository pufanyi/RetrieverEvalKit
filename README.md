# SC4020-Project1

## Development

See [docs/dev.md](docs/dev.md) for more details.

## Run Embedding

```sh
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m img_search.pipeline.embed \
  models=jina_v4 \
  tasks.batch_size=1024
```

## Flickr30k search demo

A FastAPI-based front-end for exploring the Flickr30k embeddings is available at
`img_search.frontend.flickr30k_demo`. The app loads the published
[`pufanyi/flickr30k-jina-embeddings-v4`](https://huggingface.co/datasets/pufanyi/flickr30k-jina-embeddings-v4)
vectors, builds the ANN backends used by the evaluation suite, and exposes a web
UI with text, caption-ID, and image-ID queries.

```sh
uv run uvicorn img_search.frontend.flickr30k_demo:app --host 0.0.0.0 --port 8000
```

Images are served from `data/flickr30k/images/{id}.jpg` by default. Override the
location (or filename pattern) with:

```sh
export FLICKR30K_IMAGE_ROOT="/path/to/flickr30k/images"
export FLICKR30K_IMAGE_PATTERN="{id}.jpg"  # customise if filenames differ
export FLICKR30K_CAPTION_CONFIG="text"     # optional, defaults to "text"
```

The demo automatically downloads embeddings via the Hugging Face Hub during
startup. Ensure you have access to the dataset and that the required ANN
dependencies (`faiss`, `scann`, `hnswlib`) are installed.

Total inference time: 1:25:58

```sh
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m img_search.pipeline.embed_text \
  models=jina_v4 \
  tasks.batch_size=1024
```
