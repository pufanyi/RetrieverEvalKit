import logging
from typing import override

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

from .encoder import Encoder


class _SiglipImageModule(torch.nn.Module):
    """Wraps the backbone to expose a forward expected by DataParallel."""

    def __init__(self, backbone: AutoModel):
        super().__init__()
        self.backbone = backbone

    def forward(self, **inputs):  # type: ignore[override]
        return self.backbone.get_image_features(**inputs)


class Siglip2Encoder(Encoder):
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch32-256",
        device_map: str | None = "auto",
        device: str | torch.device | None = None,
        data_parallel: bool = False,
    ):
        super().__init__("Siglip2")
        self.model_name = model_name
        self.device_map = device_map
        self._requested_device = device
        self._device = torch.device("cpu")
        self._data_parallel = data_parallel
        self._model: AutoModel | None = None
        self._parallel_model: torch.nn.Module | None = None
        self._processor: AutoProcessor | None = None

    def build(self):
        logger = logging.getLogger(__name__)

        load_kwargs: dict[str, object] = {}
        use_data_parallel = self._data_parallel and torch.cuda.device_count() > 1

        if use_data_parallel:
            load_kwargs["device_map"] = None
        elif self.device_map is not None:
            load_kwargs["device_map"] = self.device_map

        backbone = AutoModel.from_pretrained(self.model_name, **load_kwargs).eval()

        if self._requested_device is None:
            if torch.cuda.is_available():
                target_device = torch.device("cuda:0")
            else:
                target_device = torch.device("cpu")
        else:
            target_device = torch.device(self._requested_device)

        if use_data_parallel:
            backbone.to(target_device)
            parallel_model = torch.nn.DataParallel(_SiglipImageModule(backbone))
            self._parallel_model = parallel_model
            self._device = target_device
        else:
            self._parallel_model = None
            if load_kwargs.get("device_map") is None:
                backbone.to(target_device)
                self._device = target_device
            else:
                first_param = next(backbone.parameters(), None)
                inferred_device = (
                    first_param.device if first_param is not None else target_device
                )
                self._device = torch.device(inferred_device)

        if self._data_parallel and not use_data_parallel:
            logger.warning(
                "Data parallel requested but only detected %d CUDA device(s); "
                "falling back to %s.",
                torch.cuda.device_count(),
                self._device,
            )

        self._model = backbone
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    @property
    def model(self):
        if self._model is None:
            self.build()
        if self._parallel_model is not None:
            return self._parallel_model
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def processor(self):
        if self._processor is None:
            self.build()
        return self._processor

    @override
    def batch_encode(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image | str] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        model = self.model

        with torch.no_grad():
            if images:
                processed_images = [load_image(image) for image in images]
                inputs = self.processor(
                    images=processed_images, return_tensors="pt"
                ).to(self.device)
                if isinstance(model, torch.nn.DataParallel):
                    return model(**inputs)
                return model.get_image_features(**inputs)

            if texts:
                inputs = self.processor(text=texts, return_tensors="pt").to(self.device)
                return model.module.backbone.get_text_features(**inputs)

        raise ValueError("Either images or texts must be provided to batch_encode")
