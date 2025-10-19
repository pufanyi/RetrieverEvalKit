# Flickr30k Streamlit Demo

This guide covers how to launch and operate the Streamlit interface that ships with
`img_search.frontend`. The app lets you explore Flickr30k image/caption retrieval with
the bundled FAISS, ScaNN, and HNSWlib backends and the Jina v4 encoder.

## Prerequisites

- Install the project dependencies (Streamlit is included in `pyproject.toml`):
  ```bash
  uv sync --dev
  ```
- Ensure you can access the embedding datasets referenced in
  `img_search/frontend/flickr30k_demo.py`. By default the demo expects the public
  Hugging Face dataset `pufanyi/flickr30k-jina-embeddings-v4` and loads both the
  `images` and `texts` configs.
- (Optional) Download the raw Flickr30k images locally so the interface can render
  thumbnails. The demo looks under `data/flickr30k/images/{id}.jpg` unless you point it
  elsewhere (see below).

## Launching the App

Run the Streamlit entrypoint directly:

```bash
uv run streamlit run img_search/frontend/flickr30k_demo.py
```

Streamlit starts a local server (defaults to <http://localhost:8501/>). Open the URL in a
browser to access the UI. The search engine builds once and stays cached across reruns.

If you prefer to embed the app in another script, import the helper exposed by the
package:

```python
from img_search.frontend import run_demo

if __name__ == "__main__":
    run_demo()
```

## Environment Configuration

You can adapt the demo without touching code via the following environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `FLICKR30K_CAPTION_CONFIG` | Selects which split of the caption embedding dataset to load (`texts`, `text-tiny`, etc.). | `texts` (alias `text`) |
| `FLICKR30K_IMAGE_ROOT` | Filesystem directory containing the original Flickr30k images. | `data/flickr30k/images` |
| `FLICKR30K_IMAGE_PATTERN` | Filename template for images relative to the root. Use `{id}` as the placeholder. | `{id}.jpg` |

Set them when invoking Streamlit, e.g.:

```bash
FLICKR30K_IMAGE_ROOT=/mnt/flickr30k uv run streamlit run img_search/frontend/flickr30k_demo.py
```

If the image directory is available the app serves thumbnails directly from disk.
Otherwise the results panel still lists identifiers, scores, and captions.

## Using the Interface

- **Sidebar controls** let you pick the retrieval backend, adjust the number of results,
  and switch between text, image, and caption query modes. Any backend that fails to
  initialise appears in the sidebar with its error message.
- **Query form** adapts to the selected mode—enter free-form text, look up by image ID,
  or reuse caption IDs surfaced from the dataset. Text queries are embedded on demand via
  `JinaV4Encoder`.
- **Results grid** displays ranked matches with similarity metrics, associated captions,
  and optional extra annotations in an expander.

The app surfaces a quick snapshot of dataset sizes along the top and exposes five sample
captions to make it easier to try the caption-based workflow.

## Troubleshooting

- The first startup can take a while because the app downloads the embedding datasets and
  builds ANN indexes for every backend. Subsequent reloads reuse cached resources.
- ScaNN and HNSWlib are optional dependencies; if they are missing the demo still runs
  with FAISS and lists the unavailable backends in the sidebar.
- When caption/image IDs come from the public dataset they remain strings (as stored in
  Hugging Face). Double-check the ID formatting if lookups fail.
- If you tweak the Hydra configs backing the demo (see
  `img_search/frontend/flickr30k_demo.py`), restart Streamlit so the cached datasets and
  ANN indices rebuild with the new settings.

## Under the hood

The Streamlit script performs a few key steps during startup:

1. Load the gallery and caption embeddings via `EmbeddingDatasetSpec`, respecting any
   environment overrides for dataset name or local paths.
2. Build FAISS, ScaNN, and HNSWlib indices (skipping optional dependencies) and register
   them in the sidebar selector.
3. Cache image thumbnails using `st.cache_data` to avoid re-reading files on every query and
   fall back to ID/score listings if thumbnails are unavailable.

Understanding this flow makes it easier to extend the app—for example, by injecting custom
filter widgets or swapping in a different embedding dataset. Update the `EmbeddingDatasetSpec`
instances in `flickr30k_demo.py` or set the environment variables documented above to point
at your own corpora.
