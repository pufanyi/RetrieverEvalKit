from abc import ABC, abstractmethod

import torch
from PIL import Image


class Encoder(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def batch_encode(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image | str] | None = None,
        task: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Encoder is not implemented")

    @abstractmethod
    def build(self, device: str = "cuda"):
        raise NotImplementedError("Encoder is not implemented")

    def encode(
        self,
        text: str | list[str] | None = None,
        image: Image.Image | str | list[Image.Image | str] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(text, list) or isinstance(image, list):
            return self.batch_encode(texts=text, images=image, **kwargs)
        else:
            return self.batch_encode(
                texts=[text] if text else None,
                images=[image] if image else None,
                **kwargs,
            )[0]
