from typing import Any

from omegaconf import DictConfig, OmegaConf

from .dataset import ImageDataset
from .flickr30k import Flickr30kDataset
from .inquire import InquireDataset

__all__ = ["ImageDataset", "InquireDataset", "Flickr30kDataset"]

DATASETS = {
    "inquire": InquireDataset,
    "flickr30k": Flickr30kDataset,
}


def _collect_kwargs(cfg: DictConfig) -> dict[str, Any]:
    direct_kwargs: dict[str, Any] = {}
    for key in cfg:
        if key in {"dataset", "kwargs"}:
            continue
        direct_kwargs[key] = cfg[key]

    nested_kwargs = cfg.get("kwargs", None)
    if nested_kwargs is not None:
        nested_kwargs_container = OmegaConf.to_container(nested_kwargs, resolve=True)
        if isinstance(nested_kwargs_container, dict):
            direct_kwargs.update(nested_kwargs_container)

    return direct_kwargs


def get_dataset(cfg: DictConfig) -> ImageDataset:
    dataset = cfg.dataset
    kwargs = _collect_kwargs(cfg)
    if dataset not in DATASETS:
        raise ValueError(f"Dataset {dataset} not found")
    return DATASETS[dataset](**kwargs)
