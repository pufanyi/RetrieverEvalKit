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
