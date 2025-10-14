from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("lmms-lab/flickr30k")
    print(dataset)