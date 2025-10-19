"""Command-line utilities for benchmarking ANN backends side by side."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from img_search.search.faiss_search import (
    FaissIndexConfig,
    benchmark_bruteforce,
    benchmark_methods,
    faiss_gpu_count,
    faiss_supports_gpu,
)
from img_search.search.hnswlib_search import HnswlibIndexConfig, hnswlib_available
from img_search.search.scann_search import ScannIndexConfig, scann_available


@dataclass(slots=True)
class BenchmarkSettings:
    """Parameters controlling the ANN benchmark run."""

    methods: list[dict[str, Any]] = field(default_factory=lambda: [{"method": "flat"}])
    top_k: int = 10
    recall_at: list[int] = field(default_factory=lambda: [1, 5])
    use_gpu: bool | None = None
    output_path: str | None = None
    excel_output_path: str | None = None


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


def _expand_method_configs(
    methods: Sequence[dict[str, Any]],
    use_gpu: bool | None,
    *,
    gpu_available: bool,
) -> list[dict[str, Any]]:
    """Expand configured methods into concrete ANN index settings."""

    expanded: list[dict[str, Any]] = []
    for method in methods:
        override = dict(method)
        backend = str(override.get("backend", "faiss")).lower()
        override["backend"] = backend
        override.pop("use_gpu", None)
        if backend == "faiss":
            if use_gpu is None:
                expanded.append({**override, "use_gpu": False})
                if gpu_available:
                    expanded.append({**override, "use_gpu": True})
            else:
                expanded.append(
                    {**override, "use_gpu": bool(use_gpu and gpu_available)}
                )
        else:
            expanded.append({**override, "use_gpu": False})
    return expanded


def _backend_name(
    cfg: FaissIndexConfig | ScannIndexConfig | HnswlibIndexConfig | Mapping[str, Any],
) -> str:
    if isinstance(cfg, FaissIndexConfig):
        return "faiss"
    if isinstance(cfg, ScannIndexConfig):
        return "scann"
    if isinstance(cfg, HnswlibIndexConfig):
        return "hnswlib"
    return str(cfg.get("backend", "faiss")).lower()


def _filter_unavailable_backends(
    method_configs: Sequence[
        FaissIndexConfig | ScannIndexConfig | HnswlibIndexConfig | Mapping[str, Any]
    ],
) -> list[FaissIndexConfig | ScannIndexConfig | HnswlibIndexConfig | Mapping[str, Any]]:
    """Drop configurations whose optional dependencies are not installed."""

    availability = {
        "faiss": True,
        "scann": scann_available(),
        "hnswlib": hnswlib_available(),
    }
    skipped: dict[str, int] = {}
    filtered: list[
        FaissIndexConfig | ScannIndexConfig | HnswlibIndexConfig | Mapping[str, Any]
    ] = []
    for cfg in method_configs:
        backend = _backend_name(cfg)
        if availability.get(backend, False):
            filtered.append(cfg)
        else:
            skipped[backend] = skipped.get(backend, 0) + 1
    for backend, count in skipped.items():
        dependency = {
            "scann": "ScaNN",
            "hnswlib": "HNSWlib",
        }.get(backend, backend)
        logger.warning(
            "Skipping {} {} configuration(s) because {} is not installed.",
            count,
            backend,
            dependency,
        )
    return filtered


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
    query_metadata = _collect_query_metadata(
        query_dataset, config.query_dataset, len(query_ids)
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

    gpu_supported = faiss_supports_gpu()
    available_gpu_count = faiss_gpu_count() if gpu_supported else 0
    gpu_available = gpu_supported and available_gpu_count > 0
    if config.evaluation.use_gpu:
        if not gpu_supported:
            logger.warning(
                "FAISS GPU evaluation requested but the installed package lacks GPU "
                "bindings (typically `faiss-cpu`). Install `faiss-gpu` and ensure the "
                "CUDA runtime is available. Falling back to CPU-only search.",
            )
        elif available_gpu_count == 0:
            logger.warning(
                "FAISS GPU evaluation requested but no CUDA devices were detected by "
                "FAISS. Verify your drivers and visibility. Falling back to CPU-only "
                "search.",
            )

    expanded_configs = _expand_method_configs(
        config.evaluation.methods,
        config.evaluation.use_gpu,
        gpu_available=gpu_available,
    )
    method_configs = _filter_unavailable_backends(expanded_configs)

    top_k = _ensure_top_k(config.evaluation)
    logger.info(
        "Prepared {} ANN method configuration(s) (top_k={}, recall_at={})",
        len(method_configs),
        top_k,
        config.evaluation.recall_at,
    )
    logger.info(
        "Running benchmark across {} image vectors and {} query vectors",
        len(image_ids),
        len(query_ids),
    )

    detailed_rows: list[dict[str, Any]] = []
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
        if method_configs:
            method_output = benchmark_methods(
                image_vectors,
                query_vectors,
                ids=image_ids,
                method_configs=method_configs,
                top_k=top_k,
                ground_truth=ground_truth,
                recall_points=config.evaluation.recall_at,
                progress=progress,
                collect_hits=True,
                query_ids=query_ids,
                query_metadata=query_metadata,
            )
            if isinstance(method_output, tuple):
                results, method_details = method_output
            else:
                results = method_output
                method_details = []
            detailed_rows.extend(method_details)
        else:
            results = []
    logger.info(
        "Benchmark complete; collected {} result rows",
        len(results),
    )

    def _metric_value(
        cfg: FaissIndexConfig | ScannIndexConfig | HnswlibIndexConfig | dict[str, Any],
    ) -> str:
        if isinstance(cfg, FaissIndexConfig | ScannIndexConfig | HnswlibIndexConfig):
            return str(cfg.metric).lower()
        return str(cfg.get("metric", "l2")).lower()

    metrics = sorted({_metric_value(cfg) for cfg in method_configs})
    brute_output = benchmark_bruteforce(
        image_vectors,
        query_vectors,
        ids=image_ids,
        metrics=metrics,
        top_k=top_k,
        ground_truth=ground_truth,
        recall_points=config.evaluation.recall_at,
        collect_hits=True,
        query_ids=query_ids,
        query_metadata=query_metadata,
    )
    if isinstance(brute_output, tuple):
        brute_rows, brute_details = brute_output
    else:
        brute_rows = brute_output
        brute_details = []
    if brute_rows:
        logger.info("Appended {} brute-force baseline result(s)", len(brute_rows))
    results.extend(brute_rows)
    detailed_rows.extend(brute_details)

    for row in results:
        row.setdefault("num_queries", len(query_ids))
        row.setdefault("top_k", top_k)

    excel_path = _resolve_excel_path(config.evaluation)
    if excel_path is not None:
        _write_excel(results, detailed_rows, excel_path)
    return results


def _results_table(rows: list[dict[str, Any]], settings: BenchmarkSettings) -> Table:
    console_columns = [
        "backend",
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

    table = Table(title="ANN Benchmark Summary")
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


def _collect_query_metadata(
    dataset: Any, spec: QueryDatasetSpec, total_rows: int
) -> list[dict[str, Any]]:
    if total_rows <= 0:
        return []
    column_names = getattr(dataset, "column_names", None)
    if not column_names:
        return [{} for _ in range(total_rows)]

    exclude = {spec.id_column, spec.embedding_column}
    if spec.relevance_column:
        exclude.add(spec.relevance_column)

    preferred = [
        "query_id",
        "query_text",
        "split",
        "supercategory",
        "category",
        "iconic_group",
    ]
    selected = [col for col in preferred if col in column_names and col not in exclude]
    if not selected:
        return [{} for _ in range(total_rows)]

    try:
        column_values = {col: dataset[col] for col in selected}
    except Exception:  # pragma: no cover - fallback for streaming datasets
        column_values = None

    metadata: list[dict[str, Any]] = []
    if column_values is not None:
        for index in range(total_rows):
            row = {}
            for column in selected:
                values = column_values[column]
                try:
                    row[column] = values[index]
                except Exception:
                    row[column] = None
            metadata.append(row)
        return metadata

    for index in range(total_rows):  # pragma: no cover - fallback path
        sample = dataset[index]
        row = {column: sample.get(column) for column in selected}
        metadata.append(row)
    return metadata


def _resolve_excel_path(settings: BenchmarkSettings) -> Path | None:
    path_value = settings.excel_output_path
    if path_value is not None:
        if str(path_value).strip() == "":
            return None
        candidate = Path(path_value)
    elif settings.output_path:
        candidate = Path(settings.output_path).with_suffix(".xlsx")
    else:
        candidate = Path("outputs/search_eval/details.xlsx")

    if not candidate.is_absolute():
        candidate = Path(get_original_cwd()) / candidate
    return candidate


def _ordered_detail_columns(columns: Sequence[str]) -> list[str]:
    preferred = [
        "query_index",
        "query_embedding_id",
        "query_id",
        "query_text",
        "split",
        "supercategory",
        "category",
        "iconic_group",
        "backend",
        "method",
        "method_label",
        "metric",
        "use_gpu",
        "top_k",
        "rank",
        "image_id",
        "distance",
        "is_relevant",
        "relevant_ids",
        "relevant_count",
    ]
    ordered = [column for column in preferred if column in columns]
    remaining = [column for column in columns if column not in ordered]
    return ordered + sorted(remaining)


def _write_excel(
    summary_rows: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a hard dependency in practice
        logger.warning("Skipping Excel export; pandas is unavailable: {}", exc)
        return

    logger.info("Writing benchmark details workbook to {}", path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer_manager = pd.ExcelWriter(path)
    except ModuleNotFoundError as exc:  # pragma: no cover - missing Excel backend
        logger.warning(
            "Skipping Excel export; no Excel writer backend available: {}", exc
        )
        return
    except ValueError as exc:  # pragma: no cover - pandas could not construct writer
        logger.warning("Skipping Excel export; unable to create writer: {}", exc)
        return

    with writer_manager as writer:
        summary_df = pd.DataFrame(summary_rows)
        if summary_df.empty:
            summary_df = pd.DataFrame(columns=["backend", "method"])
        summary_df.to_excel(writer, sheet_name="summary", index=False)

        detail_df = pd.DataFrame(detail_rows)
        if not detail_df.empty:
            detail_df = detail_df[_ordered_detail_columns(detail_df.columns)]
        else:  # ensure consistent headers when no results
            detail_df = pd.DataFrame(
                columns=_ordered_detail_columns(
                    [
                        "query_index",
                        "query_embedding_id",
                        "backend",
                        "method",
                        "metric",
                        "rank",
                        "image_id",
                    ]
                )
            )
        detail_df.to_excel(writer, sheet_name="details", index=False)


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
