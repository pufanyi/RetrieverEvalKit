from datasets import load_dataset
from datasets import Dataset
from tqdm.rich import tqdm

if __name__ == "__main__":
    dataset = load_dataset("pufanyi/flickr30k-jina-embeddings-v4", "images", split="train")
    data = {}
    for item in tqdm(dataset):
        data[item["id"]] = item
    final_dataset = Dataset.from_list(list(data.values()))
    final_dataset.push_to_hub("pufanyi/flickr30k-jina-embeddings-v4")