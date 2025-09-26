from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from PIL import Image
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
        default_task: str | None = "retrieval",
        model_kwargs: dict[str, object] | None = None,
    ):
        super().__init__("JinaV4")
        self.model_name = model_name
        self._requested_device = device
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._default_task = default_task
        self._model_kwargs = dict(model_kwargs) if model_kwargs else {}
        self._model: SentenceTransformer | None = None

    def build(self):
        load_kwargs: dict[str, object] = {"trust_remote_code": True}
        if self._requested_device is not None:
            load_kwargs["device"] = str(self._requested_device)
        model_kwargs = dict(self._model_kwargs)
        if self._default_task is not None and "default_task" not in model_kwargs:
            model_kwargs["default_task"] = self._default_task
        if model_kwargs:
            load_kwargs["model_kwargs"] = model_kwargs
        self._model = SentenceTransformer(self.model_name, **load_kwargs)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self.build()
        return self._model

    def _prepare_image_inputs(
        self, images: Sequence[Image.Image | str]
    ) -> list[Image.Image | str]:
        prepared: list[Image.Image | str] = []
        for image in images:
            if isinstance(image, (Image.Image, str)):
                prepared.append(image)
            else:
                raise TypeError(
                    "Image inputs must be PIL.Image.Image or str (path or URL)."
                )
        return prepared

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
        call_kwargs.update(encode_kwargs)
        task_to_use = task
        if task_to_use is None:
            task_to_use = call_kwargs.pop("task", None)
        if task_to_use is None:
            task_to_use = self._default_task
        if task_to_use is not None:
            call_kwargs["task"] = task_to_use
        if prompt_name is not None and "prompt_name" not in call_kwargs:
            call_kwargs["prompt_name"] = prompt_name
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
            inputs: Sequence[str | Image.Image] = list(texts)
        else:
            inputs = self._prepare_image_inputs(images or [])

        return self._encode(
            inputs,
            task=task,
            prompt_name=prompt_name,
            encode_kwargs=dict(kwargs),
        )
