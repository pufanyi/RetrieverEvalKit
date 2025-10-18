from typing import Any

from omegaconf import DictConfig, OmegaConf

from .encoder import Encoder
from .jina_v4 import JinaV4Encoder
from .jina_v4_vllm import JinaV4VLLMEncoder
from .siglip import SiglipEncoder
from .siglip2 import Siglip2Encoder
from .siglip_vllm import SiglipVLLMEncoder

__all__ = [
    "SiglipEncoder",
    "Siglip2Encoder",
    "JinaV4Encoder",
    "JinaV4VLLMEncoder",
    "SiglipVLLMEncoder",
]


def _collect_kwargs(cfg: DictConfig) -> dict[str, Any]:
    direct_kwargs: dict[str, Any] = {}
    for key in cfg:
        if key in {"model", "kwargs"}:
            continue
        direct_kwargs[key] = cfg[key]

    nested_kwargs = cfg.get("kwargs", None)
    if nested_kwargs is not None:
        nested_kwargs_container = OmegaConf.to_container(nested_kwargs, resolve=True)
        if isinstance(nested_kwargs_container, dict):
            direct_kwargs.update(nested_kwargs_container)

    return direct_kwargs


def get_encoder(cfg: DictConfig) -> Encoder:
    model = cfg.model
    kwargs = _collect_kwargs(cfg)
    if model == "siglip":
        return SiglipEncoder(**kwargs)
    elif model == "siglip2":
        return Siglip2Encoder(**kwargs)
    elif model == "siglip_vllm":
        return SiglipVLLMEncoder(**kwargs)
    elif model == "jina_v4":
        return JinaV4Encoder(**kwargs)
    elif model == "jina_v4_vllm":
        return JinaV4VLLMEncoder(**kwargs)
    else:
        raise ValueError(f"Model {model} not found")
