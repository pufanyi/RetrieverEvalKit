from typing import override

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

from .encoder import Encoder


class Siglip2Encoder(Encoder):
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch32-256",
        device_map: str = "auto",
    ):
        super().__init__("Siglip2")
        self.model_name = model_name
        self.device_map = device_map
        self._model: AutoModel | None = None
        self._processor: AutoProcessor | None = None

    def build(self):
        self._model = AutoModel.from_pretrained(
            self.model_name, device_map=self.device_map
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self.build()
        return self._processor

    @override
    def batch_encode(self, images: list[Image.Image | str], **kwargs) -> torch.Tensor:
        images = [load_image(image) for image in images]
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs
