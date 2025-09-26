import hydra
from omegaconf import DictConfig
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from img_search.utils.logging import print_config, setup_logger

from ..data import ImageDataset, get_dataset
from ..embedding import Encoder, get_encoder


def get_models_and_datasets(
    cfg: DictConfig,
) -> tuple[list[Encoder], list[ImageDataset]]:
    models = [get_encoder(model_cfg) for model_cfg in cfg.models]
    datasets = [get_dataset(dataset_cfg) for dataset_cfg in cfg.datasets]
    return models, datasets


def embed_all(models, datasets, *, tasks_config: DictConfig):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        for model in progress.track(models, description="Processing models"):
            model.build()
            for dataset in progress.track(
                datasets, description=f"Processing datasets for {model.name}"
            ):
                dataset.build()
                for data in progress.track(
                    dataset.get_images(batch_size=tasks_config.batch_size),
                    description=f"Embedding with {model.name} on {dataset.name}",
                ):
                    yield model.name, dataset.name, model.encode(image=data)


@hydra.main(
    config_path="pkg://img_search/config", version_base=None, config_name="embed_config"
)
def main(cfg: DictConfig):
    setup_logger(cfg.logging)
    print_config(cfg)

    models, datasets = get_models_and_datasets(cfg)

    for model_name, dataset_name, embedding in embed_all(
        models, datasets, tasks_config=cfg.tasks
    ):
        print(model_name, dataset_name, embedding)


if __name__ == "__main__":
    main()
