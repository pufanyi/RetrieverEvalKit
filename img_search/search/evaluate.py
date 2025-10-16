"""Command-line utilities for benchmarking FAISS configurations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import main as hydra_main
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from img_search.data.embeddings import (
    EmbeddingDatasetSpec,
    QueryDatasetSpec,
    extract_embeddings,
    extract_relevance,
    load_embedding_dataset,
)
from img_search.search.faiss_search import FaissIndexConfig, benchmark_methods


@dataclass(slots=True)
class BenchmarkSettings:
    """Parameters controlling the FAISS benchmark run."""

    methods: list[dict[str, Any]] = field(default_factory=lambda: [{"method": "flat"}])
    top_k: int = 10
    recall_at: list[int] = field(default_factory=lambda: [1, 5])
    use_gpu: bool | None = None
    output_path: str | None = None


@dataclass(slots=True)
class SearchEvalConfig:
    """Aggregate configuration for a benchmark run."""

    image_dataset: EmbeddingDatasetSpec
    query_dataset: QueryDatasetSpec
    evaluation: BenchmarkSettings = field(default_factory=BenchmarkSettings)


def _ensure_top_k(settings: BenchmarkSettings) -> int:
    recall_max = max(settings.recall_at, default=0)
    return max(settings.top_k, recall_max)


def _describe_spec(spec: EmbeddingDatasetSpec) -> str:
    parts: list[str] = []
    if spec.load_from_disk:
        parts.append(f"load_from_disk={spec.load_from_disk}")
    else:
        if spec.dataset_name:
            parts.append(f"name={spec.dataset_name}")
        if spec.dataset_config:
            parts.append(f"config={spec.dataset_config}")
        if spec.split:
            parts.append(f"split={spec.split}")
        if spec.revision:
            parts.append(f"revision={spec.revision}")
        if spec.data_files:
            parts.append("data_files=provided")
    parts.append(f"id_column={spec.id_column}")
    parts.append(f"embedding_column={spec.embedding_column}")
    if isinstance(spec, QueryDatasetSpec) and getattr(spec, "relevance_column", None):
        parts.append(f"relevance_column={spec.relevance_column}")
    return ", ".join(parts)


def _dataset_size(dataset: Any) -> str:
    try:
        size = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return "unknown (streaming)"
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Unable to determine dataset size: {}", exc)
        return "unknown"
    return f"{size:,}"


def run_search_evaluation(config: SearchEvalConfig) -> list[dict[str, Any]]:
    """Execute the benchmark described by ``config`` and return raw rows."""

    logger.info(
        "Loading image embeddings dataset: {}",
        _describe_spec(config.image_dataset),
    )
    image_dataset = load_embedding_dataset(config.image_dataset)
    logger.info(
        "Loaded image embeddings dataset (size={})",
        _dataset_size(image_dataset),
    )

    logger.info(
        "Loading query embeddings dataset: {}",
        _describe_spec(config.query_dataset),
    )
    query_dataset = load_embedding_dataset(config.query_dataset)
    logger.info(
        "Loaded query embeddings dataset (size={})",
        _dataset_size(query_dataset),
    )

    logger.info(
        "Extracting image embeddings (id_column='{}', embedding_column='{}')",
        config.image_dataset.id_column,
        config.image_dataset.embedding_column,
    )
    logger.debug("Image dataset type: {}", type(image_dataset))
    image_ids, image_vectors = extract_embeddings(
        image_dataset,
        id_column=config.image_dataset.id_column,
        embedding_column=config.image_dataset.embedding_column,
        batch_size=config.image_dataset.read_batch_size,
        memmap_path=config.image_dataset.memmap_path,
    )
    image_dim = image_vectors.shape[1] if image_vectors.ndim == 2 else "unknown"
    logger.info(
        "Extracted image embeddings: count={} dim={}",
        len(image_ids),
        image_dim,
    )
    logger.info(
        "Extracting query embeddings (id_column='{}', embedding_column='{}')",
        config.query_dataset.id_column,
        config.query_dataset.embedding_column,
    )
    logger.debug("Query dataset type: {}", type(query_dataset))
    query_ids, query_vectors = extract_embeddings(
        query_dataset,
        id_column=config.query_dataset.id_column,
        embedding_column=config.query_dataset.embedding_column,
        batch_size=config.query_dataset.read_batch_size,
        memmap_path=config.query_dataset.memmap_path,
    )
    query_dim = query_vectors.shape[1] if query_vectors.ndim == 2 else "unknown"
    logger.info(
        "Extracted query embeddings: count={} dim={}",
        len(query_ids),
        query_dim,
    )

    ground_truth: list[Sequence[str] | str] | None = None
    if config.query_dataset.relevance_column:
        logger.info(
            "Extracting relevance labels from column '{}'",
            config.query_dataset.relevance_column,
        )
        raw_truth = extract_relevance(
            query_dataset, relevance_column=config.query_dataset.relevance_column
        )
        ground_truth = list(raw_truth)
        logger.info(
            "Loaded relevance labels for {} queries",
            len(ground_truth),
        )

    method_configs: list[FaissIndexConfig | dict[str, Any]] = []
    for method in config.evaluation.methods:
        override = dict(method)
        if config.evaluation.use_gpu is not None:
            override["use_gpu"] = config.evaluation.use_gpu
        method_configs.append(override)

    top_k = _ensure_top_k(config.evaluation)
    logger.info(
        "Prepared {} FAISS method configuration(s) (top_k={}, recall_at={})",
        len(method_configs),
        top_k,
        config.evaluation.recall_at,
    )
    logger.info(
        "Running benchmark across {} image vectors and {} query vectors",
        len(image_ids),
        len(query_ids),
    )

    progress_console = Console(stderr=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=progress_console,
        transient=True,
        disable=not progress_console.is_interactive,
    ) as progress:
        results = benchmark_methods(
            image_vectors,
            query_vectors,
            ids=image_ids,
            method_configs=method_configs,
            top_k=top_k,
            ground_truth=ground_truth,
            recall_points=config.evaluation.recall_at,
            progress=progress,
        )
    logger.info(
        "Benchmark complete; collected {} result rows",
        len(results),
    )

    for row in results:
        row.setdefault("num_queries", len(query_ids))
        row.setdefault("top_k", top_k)
    return results


def _results_table(rows: list[dict[str, Any]], settings: BenchmarkSettings) -> Table:
    console_columns = [
        "method",
        "metric",
        "use_gpu",
        "index_time",
        "search_time",
        "avg_query_time",
    ]
    include_accuracy = any(row.get("accuracy") is not None for row in rows)
    if include_accuracy:
        console_columns.append("accuracy")
    for point in sorted({int(val) for val in settings.recall_at}):
        console_columns.append(f"recall@{point}")
    console_columns.extend(["top_k", "num_queries"])

    table = Table(title="FAISS Benchmark Summary")
    for column in console_columns:
        table.add_column(column, justify="right" if "time" in column else "left")

    for row in rows:
        values = []
        for column in console_columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("-" if value is None else str(value))
        table.add_row(*values)
    return table


def _write_output(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    logger.info("Writing benchmark summary to {}", path)

    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_config(config: DictConfig) -> SearchEvalConfig:
    container = OmegaConf.to_container(config, resolve=True)
    assert isinstance(container, dict)
    image = EmbeddingDatasetSpec(**container["image_dataset"])
    query = QueryDatasetSpec(**container["query_dataset"])
    evaluation = BenchmarkSettings(**container.get("evaluation", {}))
    return SearchEvalConfig(
        image_dataset=image, query_dataset=query, evaluation=evaluation
    )


@hydra_main(version_base=None, config_path="../config/search_eval", config_name="eval")
def app(config: DictConfig) -> list[dict[str, Any]]:
    """Hydra entry-point for ``python -m img_search.search.evaluate``."""

    parsed = _parse_config(config)
    logger.info(
        "Starting search evaluation | image_dataset={} | query_dataset={}",
        _describe_spec(parsed.image_dataset),
        _describe_spec(parsed.query_dataset),
    )
    rows = run_search_evaluation(parsed)
    logger.info("Search evaluation finished")

    console = Console()
    console.print(_results_table(rows, parsed.evaluation))

    if parsed.evaluation.output_path:
        output_path = Path(parsed.evaluation.output_path)
        if not output_path.is_absolute():
            output_path = Path(get_original_cwd()) / output_path
        _write_output(rows, output_path)
        console.print(f"Saved benchmark summary to {output_path}")
    return rows


if __name__ == "__main__":
    app()
