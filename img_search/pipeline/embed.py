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

from ..data import get_dataset
from ..embedding import get_encoder


@hydra.main(
    config_path="pkg://img_search/config", version_base=None, config_name="embed_config"
)
def main(cfg: DictConfig):
    setup_logger(cfg.logging)
    print_config(cfg)
    models = []
    datasets = []

    for model_cfg in cfg.models:
        models.append(get_encoder(model_cfg))
    for dataset_cfg in cfg.datasets:
        datasets.append(get_dataset(dataset_cfg))

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
                    dataset,
                    description=f"Embedding with {model.name} on {dataset.name}",
                ):
                    model.embed(data)


if __name__ == "__main__":
    main()
