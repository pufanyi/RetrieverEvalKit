import logging
from typing import Any, override

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

from .encoder import Encoder


class _SiglipImageModule(torch.nn.Module):
    """Wraps the image forward pass for DataParallel execution."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, **inputs):  # type: ignore[override]
        return self.backbone.get_image_features(**inputs)


class SiglipEncoder(Encoder):
    """Encoder powered by `google/siglip` checkpoints."""

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device_map: str | None = "auto",
        device: str | torch.device | None = None,
        data_parallel: bool = False,
    ) -> None:
        super().__init__("Siglip")
        self.model_name = model_name
        self.device_map = device_map
        self._requested_device = device
        self._device = torch.device("cpu")
        self._data_parallel = data_parallel
        self._model: AutoModel | None = None
        self._parallel_model: torch.nn.Module | None = None
        self._processor: AutoProcessor | None = None

    def build(self) -> None:
        logger = logging.getLogger(__name__)

        load_kwargs: dict[str, Any] = {}
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
            self._parallel_model = torch.nn.DataParallel(_SiglipImageModule(backbone))
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
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self.build()
        if self._parallel_model is not None:
            return self._parallel_model
        assert self._model is not None
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def processor(self) -> AutoProcessor:
        if self._processor is None:
            self.build()
        assert self._processor is not None
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
                image_kwargs = dict(kwargs)
                image_kwargs.setdefault("return_tensors", "pt")
                inputs = self.processor(images=processed_images, **image_kwargs).to(
                    self.device
                )
                if isinstance(model, torch.nn.DataParallel):
                    return model(**inputs)
                return model.get_image_features(**inputs)

            if texts:
                text_kwargs = dict(kwargs)
                text_kwargs.setdefault("padding", True)
                text_kwargs.setdefault("truncation", True)
                text_kwargs.setdefault("return_tensors", "pt")
                inputs = self.processor(text=texts, **text_kwargs).to(self.device)
                if isinstance(model, torch.nn.DataParallel):
                    assert self._model is not None
                    return self._model.get_text_features(**inputs)
                return model.get_text_features(**inputs)

        raise ValueError("Either images or texts must be provided to batch_encode")
