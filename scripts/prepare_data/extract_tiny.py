from __future__ import annotations

import argparse
import logging

from datasets import load_dataset


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create and upload a tiny subset of a dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="pufanyi/flickr30k-jina-embeddings-v4",
        help="The Hugging Face repository ID.",
    )
    parser.add_argument(
        "--source-config",
        type=str,
        default="texts",
        help="The name of the source configuration to sample from.",
    )
    parser.add_argument(
        "--target-config",
        type=str,
        default="text-tiny",
        help="The name for the new tiny configuration.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="The number of samples to extract for the tiny subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token. If not provided, will use cached token.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to create and upload a tiny subset."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = get_args()

    logging.info(
        "Loading '%s' config from %s...", args.source_config, args.repo_id
    )
    full_ds = load_dataset(args.repo_id, name=args.source_config, token=args.token)["train"]

    logging.info(
        "Creating a tiny subset of %d samples...", args.num_samples
    )
    tiny_ds = full_ds.shuffle(seed=args.seed).select(range(args.num_samples))

    logging.info("Uploading '%s' config to %s...", args.target_config, args.repo_id)
    tiny_ds.push_to_hub(
        repo_id=args.repo_id, token=args.token, config_name=args.target_config
    )
    logging.info("✅ Successfully created and uploaded the tiny subset.")


if __name__ == "__main__":
    main()