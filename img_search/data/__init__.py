from omegaconf import DictConfig

from .dataset import ImageDataset
from .inquire import InquireDataset

__all__ = ["ImageDataset", "InquireDataset"]

DATASETS = {
    "inquire": InquireDataset,
}


def get_dataset(cfg: DictConfig) -> ImageDataset:
    dataset = cfg.dataset
    kwargs = cfg.get("kwargs", {})
    if dataset not in DATASETS:
        raise ValueError(f"Dataset {dataset} not found")
    return DATASETS[dataset](**kwargs)
