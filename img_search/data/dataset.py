from abc import ABC, abstractmethod
from collections.abc import Iterator

from PIL import Image


class ImageDataset(ABC):
    def __init__(self, name: str):
        self.name = name

    def __len__(self) -> int:
        return self.length()

    def __iter__(self) -> Iterator[Image.Image | str]:
        for batch in self.get_images(batch_size=1):
            if len(batch) != 1:
                raise ValueError(f"Batch size is not 1, but {len(batch)}")
            yield batch[0]

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_images(
        self, batch_size: int = 1
    ) -> Iterator[list[tuple[str, Image.Image | str]]]:
        raise NotImplementedError


class TextDataset(ABC):
    def __init__(self, name: str):
        self.name = name

    def __len__(self) -> int:
        return self.length()

    def __iter__(self) -> Iterator[str]:
        for batch in self.get_texts(batch_size=1):
            if len(batch) != 1:
                raise ValueError(f"Batch size is not 1, but {len(batch)}")
            _, text_value = batch[0]
            yield text_value

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_texts(self, batch_size: int = 1) -> Iterator[list[tuple[int, str]]]:
        raise NotImplementedError

    @abstractmethod
    def get_record(self, index: int):
        raise NotImplementedError
