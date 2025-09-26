from __future__ import annotations

from typing import Literal, Sequence

import torch
from PIL import Image
from transformers.image_utils import load_image
from sentence_transformers import SentenceTransformer

from .encoder import Encoder


class JinaV4Encoder(Encoder):
    """SentenceTransformer-backed encoder for jinaai/jina-embeddings-v4."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str | torch.device | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
    ):
        super().__init__("JinaV4")
        self.model_name = model_name
        self._requested_device = device
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None

    def build(self):
        load_kwargs: dict[str, object] = {"trust_remote_code": True}
        if self._requested_device is not None:
            load_kwargs["device"] = str(self._requested_device)
        self._model = SentenceTransformer(self.model_name, **load_kwargs)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self.build()
        return self._model

    def _encode(
        self,
        inputs: Sequence[str | Image.Image],
        *,
        task: str | None,
        prompt_name: Literal["query", "passage", "code"] | None,
        encode_kwargs: dict[str, object],
    ) -> torch.Tensor | list[torch.Tensor]:
        call_kwargs: dict[str, object] = {
            "convert_to_tensor": True,
            "normalize_embeddings": self._normalize_embeddings,
        }
        if self._batch_size is not None and "batch_size" not in encode_kwargs:
            call_kwargs["batch_size"] = self._batch_size
        if task is not None:
            call_kwargs["task"] = task
        if prompt_name is not None:
            call_kwargs["prompt_name"] = prompt_name
        call_kwargs.update(encode_kwargs)
        return self.model.encode(inputs, **call_kwargs)

    def batch_encode(
        self,
        texts: Sequence[str] | None = None,
        images: Sequence[Image.Image | str] | None = None,
        *,
        task: str | None = None,
        prompt_name: Literal["query", "passage", "code"] | None = None,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        if texts is not None and images is not None:
            raise ValueError("Provide either texts or images, not both.")
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided.")

        if texts is not None:
            inputs = texts
        else:
            inputs = [load_image(image) for image in images]

        return self._encode(
            inputs,
            task=task,
            prompt_name=prompt_name,
            encode_kwargs=dict(kwargs),
        )
