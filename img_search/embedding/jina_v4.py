from typing import override

import torch
from PIL import Image
from transformers import AutoModel

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
        self._model = (
            AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=self.dtype,
            )
            .eval()
            .to(self.device)
        )

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
        prompt_name: str | None = "query",
        **kwargs,
    ) -> torch.Tensor:
        if texts and images:
            raise ValueError("texts and images cannot be provided at the same time")
        if task not in self.tasks:
            raise ValueError(f"Task {task} not found, available tasks: {self.tasks}")
        if texts:
            AVAILABLE_PROMPT_NAMES = {"query", "passage", "code"}
            if prompt_name not in AVAILABLE_PROMPT_NAMES:
                raise ValueError(
                    f"Prompt name {prompt_name} not found, "
                    f"available prompt names: {AVAILABLE_PROMPT_NAMES}"
                )
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
