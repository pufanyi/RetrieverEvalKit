from .dataset import ImageDataset
from .inquire import InquireDataset

__all__ = ["ImageDataset", "InquireDataset"]

DATASETS = {
    "inquire": InquireDataset,
}


def get_dataset(dataset: str, **kwargs) -> ImageDataset:
    if dataset not in DATASETS:
        raise ValueError(f"Dataset {dataset} not found")
    return DATASETS[dataset](**kwargs)
