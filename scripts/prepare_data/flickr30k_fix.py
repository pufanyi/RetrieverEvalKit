from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    path = Path("data/flickr30k/embeddings/image.parquet")

    # Read the parquet file
    df = pd.read_parquet(path)

    print(f"Original rows: {len(df)}")
    print(f"Unique IDs: {df['id'].nunique()}")

    # Remove duplicates, keep first occurrence
    df_dedup = df.drop_duplicates(subset=["id"], keep="first")

    print(f"After deduplication: {len(df_dedup)}")

    # Save the deduplicated data
    df_dedup.to_parquet(path, index=False)
    print(f"Saved deduplicated data to {path}")
