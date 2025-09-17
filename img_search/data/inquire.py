from typing import Iterator

from PIL import Image
from datasets import load_dataset
from .dataset import ImageDataset

class InquireDataset(ImageDataset):
    def __init__(self, path: str = "evendrow/INQUIRE-Rerank"):
        self.dataset = load_dataset(path)

    def get_images(self) -> Iterator[Image.Image | str]:
        pass
