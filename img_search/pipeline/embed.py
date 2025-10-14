from pathlib import Path
import time
import logging

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from accelerate import Accelerator
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


def safe_build_model(model: Encoder, accelerator: Accelerator, max_retries: int = 3) -> None:
    """Safely build model with synchronization and retry logic."""
    logger = logging.getLogger(__name__)
    
    if accelerator.is_main_process:
        # Main process downloads the model
        for attempt in range(max_retries):
            try:
                logger.info(f"Main process building model {model.name} (attempt {attempt + 1})")
                model.build()
                logger.info(f"Main process successfully built model {model.name}")
                break
            except Exception as e:
                logger.warning(f"Main process failed to build model {model.name} on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Main process failed to build model {model.name} after {max_retries} attempts")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    # Synchronize all processes
    accelerator.wait_for_everyone()
    
    if not accelerator.is_main_process:
        # Other processes load from cache
        for attempt in range(max_retries):
            try:
                logger.info(f"Worker process {accelerator.process_index} building model {model.name} (attempt {attempt + 1})")
                model.build()
                logger.info(f"Worker process {accelerator.process_index} successfully built model {model.name}")
                break
            except Exception as e:
                logger.warning(f"Worker process {accelerator.process_index} failed to build model {model.name} on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Worker process {accelerator.process_index} failed to build model {model.name} after {max_retries} attempts")
                    raise
                time.sleep(1 + attempt)  # Staggered retry


def safe_build_dataset(dataset: ImageDataset, accelerator: Accelerator, max_retries: int = 3) -> None:
    """Safely build dataset with synchronization and retry logic."""
    logger = logging.getLogger(__name__)
    
    if accelerator.is_main_process:
        # Main process downloads the dataset
        for attempt in range(max_retries):
            try:
                logger.info(f"Main process building dataset {dataset.name} (attempt {attempt + 1})")
                dataset.build()
                logger.info(f"Main process successfully built dataset {dataset.name}")
                break
            except Exception as e:
                logger.warning(f"Main process failed to build dataset {dataset.name} on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Main process failed to build dataset {dataset.name} after {max_retries} attempts")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    # Synchronize all processes
    accelerator.wait_for_everyone()
    
    if not accelerator.is_main_process:
        # Other processes load from cache
        for attempt in range(max_retries):
            try:
                logger.info(f"Worker process {accelerator.process_index} building dataset {dataset.name} (attempt {attempt + 1})")
                dataset.build()
                logger.info(f"Worker process {accelerator.process_index} successfully built dataset {dataset.name}")
                break
            except Exception as e:
                logger.warning(f"Worker process {accelerator.process_index} failed to build dataset {dataset.name} on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Worker process {accelerator.process_index} failed to build dataset {dataset.name} after {max_retries} attempts")
                    raise
                time.sleep(1 + attempt)  # Staggered retry


def get_models_and_datasets(
    cfg: DictConfig,
) -> tuple[list[Encoder], list[ImageDataset]]:
    print(cfg.models)
    models = [get_encoder(model_cfg) for model_cfg in cfg.models]
    datasets = [get_dataset(dataset_cfg) for dataset_cfg in cfg.datasets]
    return models, datasets


def embed_all(models, datasets, *, tasks_config: DictConfig, accelerator: Accelerator):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        main_process_progress = progress if accelerator.is_main_process else None

        models_iter = (
            progress.track(models, description="Processing models")
            if accelerator.is_main_process
            else models
        )
        for model in models_iter:
            # Use safe model building with retry logic
            safe_build_model(model, accelerator)
            
            datasets_iter = (
                progress.track(
                    datasets, description=f"Processing datasets for {model.name}"
                )
                if accelerator.is_main_process
                else datasets
            )
            for dataset in datasets_iter:
                # Use safe dataset building with retry logic
                safe_build_dataset(dataset, accelerator)
                data_loader = dataset.get_images(batch_size=tasks_config.batch_size)
                data_loader = accelerator.prepare(data_loader)

                task = (
                    main_process_progress.add_task(
                        description=f"Embedding with {model.name} on {dataset.name}",
                        total=(dataset.length() + tasks_config.batch_size - 1)
                        // tasks_config.batch_size,
                        visible=accelerator.is_main_process,
                    )
                    if main_process_progress
                    else None
                )

                for data_with_ids in data_loader:
                    ids, data = zip(*data_with_ids, strict=False)
                    result = model.batch_encode(images=list(data))

                    # Gather results from all processes to the main process for writing
                    all_ids = accelerator.gather_for_metrics(list(ids))
                    all_embeddings = accelerator.gather_for_metrics(result).cpu()

                    if accelerator.is_main_process:
                        yield model.name, dataset.name, all_ids, all_embeddings

                    if main_process_progress and task is not None:
                        main_process_progress.update(task, advance=1)


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

    accelerator = Accelerator()
    models, datasets = get_models_and_datasets(cfg)

    try:
        for model_name, dataset_name, ids, embeddings in embed_all(
            models, datasets, tasks_config=cfg.tasks, accelerator=accelerator
        ):
            # Only the main process writes the file
            if not accelerator.is_main_process:
                continue

            # 2. Convert batch data to an Arrow Table
            batch_table = pa.Table.from_pydict(
                {
                    "id": ids,
                    "model_name": [model_name] * len(ids),
                    "dataset_name": [dataset_name] * len(ids),
                    "embedding": embeddings.numpy().tolist(),
                }
            )

            # 3. Initialize writer (on first write) and write data
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), batch_table.schema)
            writer.write_table(table=batch_table)

    finally:
        # 4. Ensure the writer is properly closed
        if writer:
            writer.close()
            if accelerator.is_main_process:
                print(f"✅ Embeddings successfully saved to {output_path}")


if __name__ == "__main__":
    main()
