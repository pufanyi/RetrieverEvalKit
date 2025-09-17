from collections.abc import Iterator

from datasets import load_dataset
from PIL import Image

from .dataset import ImageDataset


class InquireDataset(ImageDataset):
    def __init__(self, path: str = "evendrow/INQUIRE-Rerank"):
        self.dataset = load_dataset(path)

    def length(self) -> int:
        return len(self.dataset)

    def get_images(self) -> Iterator[Image.Image | str]:
        for sample in self.dataset:
            yield sample["image"]
