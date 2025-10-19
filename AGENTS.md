# Repository Guidelines

## Project Structure & Module Organization
- Core library lives under `img_search/`; datasets in `img_search/data/`, encoders in `img_search/embedding/`, and Milvus-lite helpers under `img_search/database/`.
- Hydra entry point is `img_search/pipeline/embed.py`; shared logging utilities stay in `img_search/utils/`.
- Configuration files sit in `img_search/config/` grouped by `models/`, `datasets/`, `tasks/`, `logging/`, and `database/`; tweak defaults via `embed_config.yaml` or inline overrides like `tasks.batch_size=128`.
- Tests live in `tests/` with embedding suites under `tests/embedding/`, dataset coverage in `tests/data/`, and sample assets in `tests/imgs/`.
- Contributor docs belong in `docs/`; generated artifacts should go to `outputs/` (keep large binaries out of Git).

## Build, Test, and Development Commands
- `uv sync --dev` installs the pinned runtime and dev-only dependencies.
- `uv run pytest -n auto` runs the full suite in parallel; narrow focus with `uv run pytest tests/embedding/test_jina.py`.
- `uv run python -m img_search.pipeline.embed` launches the embedding pipeline with Hydra defaults; add overrides such as `models=siglip datasets=inquire` for experiments.
- Prefer Accelerate for multi-GPU runs: `uv run accelerate launch img_search/pipeline/embed.py models=jina_v4 models.0.kwargs.use_accelerate=true`.
- `uv run pre-commit run --all-files` applies Ruff formatting and linting to match CI.

## Coding Style & Naming Conventions
- Target Python 3.13, 4-space indentation, and add explicit type hints for new public APIs (see `img_search/proto/embed_result.py`).
- Modules and files keep snake_case; classes use PascalCase; Hydra config filenames mirror group names.
- Let Ruff enforce `E,F,I,B,UP,A,C4`; avoid manual import sorting or stylistic tweaks unless lint flags a violation.

## Testing Guidelines
- Use pytest for all suites; rely on existing fixtures and helpers when possible.
- Gate optional dependencies with `pytest.importorskip` to keep CI stable.
- Name new test modules descriptively in snake_case; store shared utilities in `img_search/utils/` if reused.
- Run targeted checks during iteration (`uv run pytest tests/embedding -k jina`) and full parallel runs before submitting.

## Commit & Pull Request Guidelines
- Follow single-sentence commit subjects starting with a capitalized verb, ~72 characters (e.g., "Enhance dataset loader for Hydra overrides").
- PRs should summarize the change set, list relevant Hydra overrides or datasets touched, call out manual commands executed, and attach screenshots when outputs change materially.
- Reference linked issues where applicable and ensure lint/tests pass before requesting review.

## Configuration & Data Tips
- Store environment-specific secrets outside the repo; rely on Hydra config overrides or `.env` handling ignored by Git.
- Keep dataset adapters deterministic by pinning source revisions and documenting assumptions in module docstrings.