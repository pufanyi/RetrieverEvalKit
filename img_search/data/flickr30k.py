from collections.abc import Iterator

from datasets import load_dataset
from PIL import Image

from .dataset import ImageDataset


class Flickr30kDataset(ImageDataset):
    def __init__(self, path: str = "lmms-lab/flickr30k", split="test"):
        super().__init__("Flickr30k")
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
            _id = sample["filename"].split(".")[0]
            current_list.append((_id, sample["image"]))
            if len(current_list) == batch_size:
                yield current_list
                current_list = []
        if current_list:
            yield current_list
