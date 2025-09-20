from typing import Literal, override

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from .encoder import Encoder


class JinaV4Encoder(Encoder):
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self.tasks = {"retrieval", "text-matching", "code"}

    def build(self):
        self._model = SentenceTransformer(self.model_name, trust_remote_code=True)

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    @override
    def batch_encode(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image | str] | None = None,
        task: str | None = "retrieval",
        prompt_name: Literal["query", "passage", "code"] | None = "query",
        **kwargs,
    ) -> torch.Tensor:
        if texts and images:
            raise ValueError("texts and images cannot be provided at the same time")
        if task not in self.tasks:
            raise ValueError(f"Task {task} not found, available tasks: {self.tasks}")
        if texts:
            text_embeddings = self.model.encode(
                sentences=texts,
                task=task,
                prompt_name=prompt_name,
            )
            return text_embeddings
        elif images:
            image_embeddings = self.model.encode(
                sentences=images,
                task=task,
            )
            return image_embeddings
        else:
            raise ValueError("Please provide either texts or images")
