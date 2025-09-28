# Repository Guidelines

## Project Structure & Module Organization
- Core package lives under `img_search/`. Dataset adapters sit in `img_search/data/`, encoder wrappers in `img_search/embedding/`, and Milvus-lite helpers for persistence live in `img_search/database/`.
- Hydra entry logic stays in `img_search/pipeline/embed.py`; shared logging utilities live in `img_search/utils/`.
- Configuration is maintained in `img_search/config/` with Hydra groups for `models/`, `datasets/`, `tasks/`, `logging/`, and `database/`. Adjust defaults through `embed_config.yaml` or inline overrides such as `tasks.batch_size=128`.
- Smoke and regression tests reside in `tests/` (embedding coverage under `tests/embedding/`, dataset checks in `tests/data/`). Sample assets for quick validation are under `tests/imgs/`.
- Contributor-focused docs stay in `docs/`, while generated artifacts should be written to `outputs/` (avoid committing large binaries).

## Build, Test, and Development Commands
- `uv sync --dev` installs the pinned runtime plus development dependencies.
- `uv run pytest -n auto` executes the test suite in parallel; target individual files with `uv run pytest tests/embedding/test_jina.py` when iterating locally.
- `uv run python -m img_search.pipeline.embed` launches the embedding pipeline with Hydra defaults. Override components inline (`uv run python -m img_search.pipeline.embed models=siglip2 datasets=inquire tasks.batch_size=128`).
- For multi-GPU runs, prefer Accelerate: `uv run accelerate launch img_search/pipeline/embed.py models=jina_v4 models.0.kwargs.use_accelerate=true` plus any additional overrides.
- `uv run pre-commit run --all-files` applies the Ruff-based lint and formatting suite expected by CI.

## Coding Style & Naming Conventions
- Target Python 3.13 with 4-space indentation and explicit type hints for new public APIs (see `img_search/proto/embed_result.py`).
- Modules and files stay snake_case; classes use PascalCase; Hydra configs mirror their group names.
- Let Ruff (configured for `E,F,I,B,UP,A,C4`) handle import ordering and common pitfalls—avoid manual restyling unless the linter flags it.

## Testing Guidelines
- Prefer pytest for coverage. Add new cases under `tests/`, reusing fixtures or helpers where reasonable, and rely on `pytest.importorskip` when optional packages (e.g., torch, PIL) are required.
- Keep new test modules descriptive (snake_case). Centralize reusable helpers in `img_search/utils/` when they benefit multiple suites.
- Use `uv run pytest tests/embedding -k <encoder>` or targeted file runs to validate integrations before submitting PRs.

## Commit & Pull Request Guidelines
- Follow the existing log: single-sentence subjects starting with a capitalized verb under ~72 characters (e.g., "Enhance dataset loader for Hydra overrides").
- For pull requests, summarize the change, list any Hydra overrides or datasets touched, note manual test commands executed, and attach screenshots only when outputs materially change.
