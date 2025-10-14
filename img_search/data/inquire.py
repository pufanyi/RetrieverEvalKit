from collections.abc import Iterator

from datasets import load_dataset
from PIL import Image

from .dataset import ImageDataset


class InquireDataset(ImageDataset):
    def __init__(self, path: str = "evendrow/INQUIRE-Rerank", split="test"):
        super().__init__("INQUIRE")
        self.dataset_path = path
        self._dataset = None
        self.split = split

    def build(self):
        self._dataset = load_dataset(self.dataset_path, split=self.split)

    @property
    def dataset(self):
        if not self._dataset:
            self.build()
        return self._dataset

    def length(self) -> int:
        return len(self.dataset)

    def get_images(
        self, batch_size: int = 1
    ) -> Iterator[list[tuple[str, Image.Image | str]]]:
        current_list = []
        for sample in self.dataset:
            current_list.append((sample["caption"], sample["image"]))
            if len(current_list) == batch_size:
                yield current_list
                current_list = []
        if current_list:
            yield current_list
