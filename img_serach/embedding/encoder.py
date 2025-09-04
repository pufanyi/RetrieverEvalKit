from abc import ABC, abstractmethod

import torch
from PIL import Image


class Encoder(ABC):
    @abstractmethod
    def batch_encode(self, images: list[Image.Image | str]) -> torch.Tensor:
        raise NotImplementedError("Encoder is not implemented")

    @abstractmethod
    def build(self, device: str = "cuda"):
        raise NotImplementedError("Encoder is not implemented")

    def encode(self, image: Image.Image | str) -> torch.Tensor:
        return self.batch_encode([image])[0]
