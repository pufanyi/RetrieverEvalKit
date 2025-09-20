from collections.abc import Iterator

from datasets import load_dataset
from PIL import Image

from .dataset import ImageDataset


class InquireDataset(ImageDataset):
    def __init__(self, path: str = "evendrow/INQUIRE-Rerank", split="test"):
        super().__init__("INQUIRE")
        self.dataset_path = path
        self._dataset = None
        self.split=split

    def build(self):
        self._dataset = load_dataset(self.dataset_path, split=self.split)
    
    @property
    def dataset(self):
        if not self._dataset:
            self.build()
        return self._dataset

    def length(self) -> int:
        return len(self.dataset)

    def get_images(self) -> Iterator[Image.Image | str]:
        for sample in self.dataset:
            yield sample["image"]
