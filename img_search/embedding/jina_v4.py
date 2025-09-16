from typing import override

import torch
from PIL import Image
from transformers import AutoModel

from .encoder import Encoder


class JinaV4Encoder(Encoder):
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._model = None
        self.tasks = {"retrieval", "text-matching", "code"}

    def build(self):
        self._model = AutoModel.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        ).eval()

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    @override
    def batch_encode(
        self, texts: list[str] | None = None, images: list[Image.Image | str] | None = None, task: str | None = "retrieval", prompt_name: str | None = "query", **kwargs
    ) -> torch.Tensor:
        if texts and images:
            raise ValueError("texts and images cannot be provided at the same time")
        if task not in self.tasks:
            raise ValueError(f"Task {task} not found, available tasks: {self.tasks}")
        if texts:
            AVAILABLE_PROMPT_NAMES = {"query", "passage", "code"}
            text_embeddings = self.model.encode_text(
                texts=texts,
                task=task,
                prompt_name=prompt_name,
            )
            return text_embeddings
        elif images:
            image_embeddings = self.model.encode_image(
                images=images,
                task=task,
            )
            return image_embeddings
        else:
            raise ValueError("Please provide either texts or images")
