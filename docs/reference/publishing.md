# Publishing Embeddings to the Hub

Once image and caption vectors are generated, `scripts/upload_to_hub.py` pushes
the Parquet outputs to a Hugging Face dataset repository. The script creates the
target repo if needed and uploads `images` and `texts` configs in one call.【F:scripts/upload_to_hub.py†L1-L68】

Follow the checklist below when preparing a release:

1. Ensure both image and caption embedding tables are present locally with
   consistent identifier columns.
2. Run the upload script with the desired repository name and authentication
   token, for example:

   ```bash
   uv run python scripts/upload_to_hub.py \
     --repo your-org/your-embeddings \
     --image-path outputs/inquire_images.parquet \
     --text-path outputs/inquire_captions.parquet
   ```

3. Confirm the Hugging Face dataset card documents the Hydra overrides used to
   generate the embeddings so others can reproduce them.
4. Update downstream configs (e.g. Streamlit demo, ANN presets) to point to the
   published dataset URLs.
