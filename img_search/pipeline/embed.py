from pathlib import Path

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from omegaconf import DictConfig

from img_search.utils.logging import print_config, setup_logger
from ..data import ImageDataset, get_dataset
from ..embedding import Encoder, get_encoder


def get_models_and_datasets(
    cfg: DictConfig,
) -> tuple[list[Encoder], list[ImageDataset]]:
    print(cfg.models)
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
                for data_with_ids in progress.track(
                    dataset.get_images(batch_size=tasks_config.batch_size),
                    description=f"Embedding with {model.name} on {dataset.name}",
                    total=(dataset.length() + tasks_config.batch_size - 1)
                    // tasks_config.batch_size,
                ):
                    # import pdb; pdb.set_trace()
                    ids, data = zip(*data_with_ids)
                    result = model.encode(image=list(data))
                    yield model.name, dataset.name, ids, result


@hydra.main(
    config_path="pkg://img_search/config", version_base=None, config_name="embed_config"
)
def main(cfg: DictConfig):
    setup_logger(cfg.logging)
    print_config(cfg)

    # 1. Prepare output file and writer
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None

    models, datasets = get_models_and_datasets(cfg)

    try:
        for model_name, dataset_name, ids, embeddings in embed_all(
            models, datasets, tasks_config=cfg.tasks
        ):
            # 2. Convert batch data to an Arrow Table
            batch_table = pa.Table.from_pydict(
                {
                    "id": ids,
                    "model_name": [model_name] * len(ids),
                    "dataset_name": [dataset_name] * len(ids),
                    "embedding": embeddings.tolist(),
                }
            )

            # 3. Initialize writer (on first write) and write data
            if writer is None:
                writer = pq.ParquetWriter(output_path, batch_table.schema)
            writer.write_table(table=batch_table)

    finally:
        # 4. Ensure the writer is properly closed
        if writer:
            writer.close()
            print(f"✅ Embeddings successfully saved to {output_path}")


if __name__ == "__main__":
    main()
