from abc import ABC, abstractmethod
from collections.abc import Iterator

from PIL import Image


class ImageDataset(ABC):
    def __len__(self) -> int:
        return self.length()

    def __iter__(self) -> Iterator[Image.Image | str]:
        return self.get_images()

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_images(self) -> Iterator[Image.Image | str]:
        raise NotImplementedError
