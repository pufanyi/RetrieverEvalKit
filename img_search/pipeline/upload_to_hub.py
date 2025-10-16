from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload image and text embeddings to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="pufanyi/flickr30k-jina-embeddings-v4",
        help="The Hugging Face repository ID to upload the dataset to.",
    )
    parser.add_argument(
        "--image-parquet",
        type=Path,
        default="data/flickr30k/embeddings/image.parquet",
        help="Path to the Parquet file containing image embeddings.",
    )
    parser.add_argument(
        "--text-parquet",
        type=Path,
        default="data/flickr30k/embeddings/text.parquet",
        help="Path to the Parquet file containing text embeddings.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token. If not provided, will use cached token.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, the repository will be created as private.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to load data and upload to Hugging Face Hub."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = get_args()

    if not args.image_parquet.exists():
        logging.error("Image embeddings file not found: %s", args.image_parquet)
        return
    if not args.text_parquet.exists():
        logging.error("Text embeddings file not found: %s", args.text_parquet)
        return

    logging.info("Creating repository %s on Hugging Face Hub...", args.repo_id)
    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=args.private
    )

    logging.info("Loading datasets from Parquet files...")
    image_ds = load_dataset("parquet", data_files={"train": str(args.image_parquet)})[
        "train"
    ]
    text_ds = load_dataset("parquet", data_files={"train": str(args.text_parquet)})[
        "train"
    ]

    image_ds.push_to_hub(repo_id=args.repo_id, token=args.token, config_name="images")
    text_ds.push_to_hub(repo_id=args.repo_id, token=args.token, config_name="texts")

    logging.info("✅ Successfully uploaded datasets to %s.", args.repo_id)


if __name__ == "__main__":
    main()
