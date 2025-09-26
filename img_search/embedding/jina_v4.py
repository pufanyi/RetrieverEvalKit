import logging
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, override

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from .encoder import Encoder


class JinaV4Encoder(Encoder):
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        data_parallel: bool = False,
    ):
        super().__init__("JinaV4")
        self.model_name = model_name
        self.dtype = dtype
        self._requested_device = device
        self._device = torch.device("cpu")
        self._data_parallel = data_parallel
        self._model: SentenceTransformer | None = None
        self._replica_models: list[SentenceTransformer] = []
        self.tasks = {"retrieval", "text-matching", "code"}

    def build(self):
        logger = logging.getLogger(__name__)

        base_model = SentenceTransformer(self.model_name, trust_remote_code=True)
        base_model = base_model.to(dtype=self.dtype)
        base_model.eval()

        if self._requested_device is None:
            if torch.cuda.is_available():
                target_device = torch.device("cuda:0")
            else:
                target_device = torch.device("cpu")
        else:
            target_device = torch.device(self._requested_device)

        use_data_parallel = self._data_parallel and torch.cuda.device_count() > 1

        if use_data_parallel:
            device_indices = list(range(torch.cuda.device_count()))
            if target_device.type == "cuda" and target_device.index in device_indices:
                device_indices.remove(target_device.index)
                device_indices.insert(0, target_device.index)

            state_dict = base_model.state_dict()
            replicas: list[SentenceTransformer] = []
            for order, index in enumerate(device_indices):
                if order == 0:
                    replica = base_model
                else:
                    replica = SentenceTransformer(
                        self.model_name, trust_remote_code=True
                    )
                    replica.load_state_dict(state_dict)
                    replica = replica.to(dtype=self.dtype)
                    replica.eval()
                replica.to(torch.device(f"cuda:{index}"))
                replicas.append(replica)

            self._model = replicas[0]
            self._replica_models = replicas
            primary_index = device_indices[0] if device_indices else 0
            self._device = torch.device(f"cuda:{primary_index}")
        else:
            base_model.to(target_device)
            self._model = base_model
            self._replica_models = [base_model]
            self._device = target_device

        if self._data_parallel and not use_data_parallel:
            logger.warning(
                "Data parallel requested but only detected %d CUDA device(s); falling back to %s.",
                torch.cuda.device_count(),
                self._device,
            )

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @override
    def batch_encode(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image | str] | None = None,
        task: str | None = "retrieval",
        prompt_name: Literal["query", "passage", "code"] | None = "query",
        **kwargs,
    ) -> torch.Tensor:
        # logger.info(f"Encoding {len(texts)} texts and {len(images)} images")
        if texts and images:
            raise ValueError("texts and images cannot be provided at the same time")
        if task not in self.tasks:
            raise ValueError(f"Task {task} not found, available tasks: {self.tasks}")
        if texts:
            return self._encode_texts(texts, task=task, prompt_name=prompt_name)
        elif images:
            return self._encode_images(images, task=task)
        else:
            raise ValueError("Please provide either texts or images")

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        task: str,
        prompt_name: Literal["query", "passage", "code"] | None,
    ) -> torch.Tensor:
        def run(model: SentenceTransformer, batch: Sequence[str]) -> torch.Tensor:
            embeddings = model.encode(
                sentences=list(batch),
                task=task,
                prompt_name=prompt_name,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=str(getattr(model, "_target_device", self.device)),
            )
            return embeddings.detach().cpu()

        return self._run_parallel(texts, run)

    def _encode_images(
        self,
        images: Sequence[Image.Image | str],
        *,
        task: str,
    ) -> torch.Tensor:
        def run(model: SentenceTransformer, batch: Sequence[Image.Image | str]) -> torch.Tensor:
            embeddings = model.encode(
                sentences=list(batch),
                task=task,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=str(getattr(model, "_target_device", self.device)),
            )
            return embeddings.detach().cpu()

        return self._run_parallel(images, run)

    def _run_parallel(self, items: Sequence, worker_fn):
        if not items:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return torch.empty((0, embedding_dim))

        models = self._replica_models if self._replica_models else [self.model]
        active_models = models[: min(len(items), len(models))]

        if len(active_models) == 1:
            return worker_fn(active_models[0], items)

        slices = self._split_indices(len(items), len(active_models))
        results: list[tuple[int, torch.Tensor]] = []

        with ThreadPoolExecutor(max_workers=len(slices)) as executor:
            futures = []
            for model, (start, end) in zip(active_models, slices):
                batch = items[start:end]
                futures.append(
                    (start, executor.submit(worker_fn, model, batch))
                )

            for start, future in futures:
                results.append((start, future.result()))

        results.sort(key=lambda item: item[0])
        return torch.cat([tensor for _, tensor in results], dim=0)

    @staticmethod
    def _split_indices(length: int, parts: int) -> list[tuple[int, int]]:
        chunk = max(1, math.ceil(length / parts))
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < length:
            end = min(start + chunk, length)
            ranges.append((start, end))
            start = end
        return ranges
