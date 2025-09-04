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
        self.model_name = model_name
        self.model: AutoModel | None = None
        self.processor: AutoProcessor | None = None
        self.device_map = device_map

    def build(self):
        self.model = AutoModel.from_pretrained(
            self.model_name, device_map=self.device_map
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def batch_encode(self, images: list[Image.Image | str]) -> torch.Tensor:
        images = [load_image(image) for image in images]
        inputs = self.processor(images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
