import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    output_path = Path("data/flickr30k")
    dataset = load_dataset("lmms-lab/flickr30k")
    image_path = output_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "captions.jsonl", "w") as f:
        for item in tqdm(dataset["test"], desc="Processing dataset"):
            file_name = item["filename"]
            item["image"].save(image_path / file_name)
            caption = {"caption": item["caption"], "image": file_name}
            f.write(json.dumps(caption) + "\n")
