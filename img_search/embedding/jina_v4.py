from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from .encoder import Encoder

if TYPE_CHECKING:
    from accelerate import Accelerator


class JinaV4Encoder(Encoder):
    """SentenceTransformer-backed encoder with optional Accelerate data parallelism."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str | torch.device | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
        default_task: str | None = "retrieval",
        model_kwargs: dict[str, object] | None = None,
        use_accelerate: bool = False,
        accelerator_kwargs: dict[str, object] | None = None,
    ):
        super().__init__("JinaV4")
        self.model_name = model_name
        self._requested_device = device
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._default_task = default_task
        self._model_kwargs = dict(model_kwargs) if model_kwargs else {}
        self._use_accelerate = use_accelerate
        self._accelerator_kwargs = dict(accelerator_kwargs) if accelerator_kwargs else {}
        self._accelerator: Accelerator | None = None
        self._model: SentenceTransformer | None = None

    def build(self):
        accelerator = None
        if self._use_accelerate:
            try:
                from accelerate import Accelerator
            except ImportError as exc:  # pragma: no cover - handled at runtime
                raise RuntimeError(
                    "Accelerate is required when `use_accelerate=True`. "
                    "Install it via `pip install accelerate` or add it to your dependencies."
                ) from exc
            accelerator = Accelerator(**self._accelerator_kwargs)
            self._accelerator = accelerator

        load_kwargs: dict[str, object] = {"trust_remote_code": True}
        if accelerator is not None:
            load_kwargs["device"] = str(accelerator.device)
        elif self._requested_device is not None:
            load_kwargs["device"] = str(self._requested_device)

        model_kwargs = dict(self._model_kwargs)
        if self._default_task is not None and "default_task" not in model_kwargs:
            model_kwargs["default_task"] = self._default_task
        if model_kwargs:
            load_kwargs["model_kwargs"] = model_kwargs

        self._model = SentenceTransformer(self.model_name, **load_kwargs)

    @property
    def accelerator(self) -> Accelerator | None:
        return self._accelerator

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

    def _encode_with_accelerate(
        self,
        inputs: list[str | Image.Image],
        call_kwargs: dict[str, object],
    ) -> torch.Tensor:
        accelerator = self._accelerator
        if accelerator is None:
            raise RuntimeError("Accelerator has not been initialized.")

        accelerator.wait_for_everyone()
        per_process_inputs = accelerator.split_between_processes(inputs)

        if per_process_inputs:
            local_embeddings = self.model.encode(per_process_inputs, **call_kwargs)
        else:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            local_embeddings = torch.empty(
                (0, embedding_dim), device=accelerator.device, dtype=torch.float32
            )

        if not isinstance(local_embeddings, torch.Tensor):
            local_embeddings = torch.as_tensor(local_embeddings)

        local_embeddings = local_embeddings.to(accelerator.device)
        local_count = torch.tensor([local_embeddings.shape[0]], device=accelerator.device)
        counts = accelerator.gather(local_count)
        padded_local = accelerator.pad_across_processes(local_embeddings, dim=0)
        gathered = accelerator.gather(padded_local)

        max_chunk = padded_local.shape[0]
        counts_list = counts.cpu().tolist()
        chunks = torch.split(gathered, max_chunk, dim=0)
        trimmed = [chunk[:count] for chunk, count in zip(chunks, counts_list) if count > 0]
        if trimmed:
            embeddings = torch.cat(trimmed, dim=0)
        else:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            embeddings = torch.empty((0, embedding_dim), dtype=gathered.dtype)

        accelerator.wait_for_everyone()
        return embeddings.cpu()

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

        if task is not None:
            call_kwargs["task"] = task
        elif "task" not in call_kwargs and self._default_task is not None:
            call_kwargs["task"] = self._default_task

        if prompt_name is not None and "prompt_name" not in call_kwargs:
            call_kwargs["prompt_name"] = prompt_name

        accelerator = self._accelerator
        if accelerator is not None and accelerator.num_processes > 1:
            return self._encode_with_accelerate(list(inputs), call_kwargs)

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
