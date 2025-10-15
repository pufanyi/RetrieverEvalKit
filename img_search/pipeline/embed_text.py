from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
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

from ..data import CaptionRecord, TextDataset, get_text_dataset
from ..embedding import Encoder, get_encoder


def safe_build_model(
    model: Encoder, accelerator: Accelerator, max_retries: int = 3
) -> None:
    """Safely build model with synchronization and retry logic."""
    logger = logging.getLogger(__name__)

    if accelerator.is_main_process:
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Main process building model %s (attempt %d)",
                    model.name,
                    attempt + 1,
                )
                model.build()
                logger.info("Main process successfully built model %s", model.name)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Main process failed to build model %s on attempt %d: %s",
                    model.name,
                    attempt + 1,
                    exc,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "Main process failed to build model %s after %d attempts",
                        model.name,
                        max_retries,
                    )
                    raise
                time.sleep(2**attempt)

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Worker process %d building model %s (attempt %d)",
                    accelerator.process_index,
                    model.name,
                    attempt + 1,
                )
                model.build()
                logger.info(
                    "Worker process %d successfully built model %s",
                    accelerator.process_index,
                    model.name,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Worker process %d failed to build model %s on attempt %d: %s",
                    accelerator.process_index,
                    model.name,
                    attempt + 1,
                    exc,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "Worker process %d failed to build model %s after %d attempts",
                        accelerator.process_index,
                        model.name,
                        max_retries,
                    )
                    raise
                time.sleep(1 + attempt)


def safe_build_dataset(
    dataset: TextDataset, accelerator: Accelerator, max_retries: int = 3
) -> None:
    """Safely build text dataset with synchronization and retry logic."""
    logger = logging.getLogger(__name__)

    if accelerator.is_main_process:
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Main process building dataset %s (attempt %d)",
                    dataset.name,
                    attempt + 1,
                )
                dataset.build()
                logger.info("Main process successfully built dataset %s", dataset.name)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Main process failed to build dataset %s on attempt %d: %s",
                    dataset.name,
                    attempt + 1,
                    exc,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "Main process failed to build dataset %s after %d attempts",
                        dataset.name,
                        max_retries,
                    )
                    raise
                time.sleep(2**attempt)

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Worker process %d building dataset %s (attempt %d)",
                    accelerator.process_index,
                    dataset.name,
                    attempt + 1,
                )
                dataset.build()
                logger.info(
                    "Worker process %d successfully built dataset %s",
                    accelerator.process_index,
                    dataset.name,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Worker process %d failed to build dataset %s on attempt %d: %s",
                    accelerator.process_index,
                    dataset.name,
                    attempt + 1,
                    exc,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        (
                            "Worker process %d failed to build dataset %s after %d "
                            "attempts"
                        ),
                        accelerator.process_index,
                        dataset.name,
                        max_retries,
                    )
                    raise
                time.sleep(1 + attempt)


def _normalize_embeddings(
    embeddings: torch.Tensor | Iterable[torch.Tensor],
) -> torch.Tensor:
    if isinstance(embeddings, torch.Tensor):
        return embeddings
    tensors = [
        tensor.detach() if tensor.requires_grad else tensor for tensor in embeddings
    ]
    return torch.stack(tensors, dim=0)


def embed_all(
    models: list[Encoder],
    dataset: TextDataset,
    *,
    tasks_config: DictConfig,
    accelerator: Accelerator,
):
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
            safe_build_model(model, accelerator)
            safe_build_dataset(dataset, accelerator)

            task = (
                main_process_progress.add_task(
                    description=f"Embedding captions with {model.name}",
                    total=(dataset.length() + tasks_config.batch_size - 1)
                    // tasks_config.batch_size,
                    visible=accelerator.is_main_process,
                )
                if main_process_progress
                else None
            )

            for batch in dataset.get_texts(batch_size=tasks_config.batch_size):
                indices, texts = zip(*batch, strict=False)
                embeddings = model.batch_encode(
                    texts=list(texts), batch_size=tasks_config.batch_size
                )
                embeddings_tensor = _normalize_embeddings(embeddings)
                embeddings_tensor = embeddings_tensor.to(accelerator.device)

                index_tensor = torch.tensor(
                    indices, device=accelerator.device, dtype=torch.long
                )

                all_indices = accelerator.gather_for_metrics(index_tensor)
                all_embeddings = accelerator.gather_for_metrics(embeddings_tensor)

                if accelerator.is_main_process:
                    gathered_indices = all_indices.tolist()
                    records: list[CaptionRecord] = [
                        dataset.get_record(idx) for idx in gathered_indices
                    ]
                    yield model.name, dataset.name, records, all_embeddings.cpu()

                if main_process_progress and task is not None:
                    main_process_progress.update(task, advance=1)


@hydra.main(
    config_path="pkg://img_search/config",
    version_base=None,
    config_name="embed_text_config",
)
def main(cfg: DictConfig):
    setup_logger(cfg.logging)
    print_config(cfg)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None

    accelerator = Accelerator()
    models = [get_encoder(model_cfg) for model_cfg in cfg.models]
    dataset = get_text_dataset(cfg.text_dataset)

    try:
        for model_name, dataset_name, records, embeddings in embed_all(
            models,
            dataset,
            tasks_config=cfg.tasks,
            accelerator=accelerator,
        ):
            if not accelerator.is_main_process:
                continue

            batch_table = pa.Table.from_pydict(
                {
                    "id": [record.caption_id for record in records],
                    "index": [record.index for record in records],
                    "image": [record.image for record in records],
                    "image_id": [record.image_id for record in records],
                    "caption": [record.caption for record in records],
                    "model_name": [model_name] * len(records),
                    "dataset_name": [dataset_name] * len(records),
                    "embedding": embeddings.numpy().tolist(),
                }
            )

            if writer is None:
                writer = pq.ParquetWriter(str(output_path), batch_table.schema)
            writer.write_table(table=batch_table)

    finally:
        if writer is not None:
            writer.close()
            if accelerator.is_main_process:
                print(f"✅ Caption embeddings successfully saved to {output_path}")


if __name__ == "__main__":
    main()
