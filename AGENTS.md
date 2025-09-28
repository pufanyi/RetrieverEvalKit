# Repository Guidelines

## Project Structure & Module Organization
- Core package lives under `img_search/`; dataset adapters sit in `img_search/data/`, encoder wrappers in `img_search/embedding/`, and Hydra entry logic in `img_search/pipeline/embed.py`.
- Configuration is maintained in `img_search/config/` with Hydra groups for `models/`, `datasets/`, and `logging/`; adjust defaults through `embed_config.yaml` or inline overrides.
- Shared helpers reside in `img_search/utils/`; contributor notes stay in `docs/`, while generated artifacts drop to `outputs/` (keep large binaries out of Git).
- Smoke tests live in `tests/embedding/`, and sample assets are under `tests/imgs/` for quick encoder validation.

## Build, Test, and Development Commands
- `uv sync --dev` installs runtime plus development dependencies pinned for this project.
- `uv run python -m img_search.pipeline.embed` launches the embedding pipeline using the Hydra defaults; override modules inline, e.g. `uv run python -m img_search.pipeline.embed models=siglip2 datasets=inquire`.
- `uv run python tests/embedding/jina.py` (or `siglip2.py`) executes smoke tests against the configured encoder wrappers.
- `uv run pre-commit run --all-files` applies the Ruff-based lint and formatting suite that CI expects.

## Coding Style & Naming Conventions
- Target Python 3.13 with 4-space indentation and prefer explicit type hints, especially for new public APIs (see `img_search/proto/embed_result.py`).
- Modules and files stay snake_case; classes use PascalCase; Hydra configs mirror their group names.
- Let Ruff (configured for `E,F,I,B,UP,A,C4`) handle import ordering and common pitfalls—avoid manual restyling unless the linter flags it.

## Testing Guidelines
- Smoke coverage currently relies on the scripts in `tests/embedding/`; extend with pytest-style tests under `tests/` when adding new behaviours.
- Name new test files descriptively in snake_case and centralize reusable fixtures or helpers in `img_search/utils/`.
- Prefer `uv run python tests/embedding/<encoder>.py` to confirm integrations before submitting PRs.

## Commit & Pull Request Guidelines
- Follow the existing log: single-sentence subjects starting with a capitalized verb under ~72 characters (e.g., "Enhance dataset loader for Hydra overrides").
- For pull requests, summarize the change, list any Hydra overrides or datasets touched, note manual test commands executed, and attach screenshots only when outputs materially change.
