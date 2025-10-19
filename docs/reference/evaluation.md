# ANN Harness and Embedding Datasets

This reference explains how embedding artifacts are represented on disk and how
the ANN benchmarking CLI consumes them to compute recall and throughput.

## Working with embedding datasets

Use `img_search.data.embeddings.EmbeddingDatasetSpec` to describe corpus
locations for both the ANN harness and the Streamlit demo. The helper handles Hub
datasets or local `load_from_disk` directories, supports chunked iteration, and
can materialise embeddings to NumPy memmaps when datasets exceed RAM.

`extract_embeddings` returns ID lists and `float32` matrices ready for ANN
indexing. When provided with a `memmap_path`, vectors are streamed to disk so
multi-million row corpora fit within bounded memory.

## Evaluating ANN backends

`img_search.search.evaluate` is the primary CLI for measuring retrieval quality:

1. Load gallery and query datasets using `EmbeddingDatasetSpec` or
   `QueryDatasetSpec` definitions.
2. Expand configured methods (FAISS, ScaNN, HNSWlib) while respecting optional
   dependency availability.
3. Build indices, compute recall/throughput, and display results in a Rich table
   while optionally exporting CSV/Excel outputs.

Preset YAML files under `img_search/config/search_eval/` include ready-to-run
combinations for Flickr30k and INQUIRE embeddings. Toggle GPU execution or tweak
backend parameters at runtime with Hydra overrides (`evaluation.use_gpu=true`,
`evaluation.methods=[...]`).
