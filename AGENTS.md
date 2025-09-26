# Repository Guidelines

## Project Structure & Module Organization
- `img_search/` is the core package. `data/` contains dataset adapters, `embedding/` defines encoder wrappers, `pipeline/embed.py` runs the Hydra-driven embedding loop, and `utils/` holds logging helpers.
- `img_search/config/` stores Hydra configs: `models/`, `datasets/`, and `logging/` choose runtime variants via `embed_config.yaml`.
- `tests/embedding/` provides runnable smoke scripts; `tests/imgs/` stores sample assets.
- `docs/` collects contributor notes, and `outputs/` is the default drop-off for generated artifacts; keep large binaries out of version control.

## Build, Test, and Development Commands
- `uv sync --dev` installs runtime and dev dependencies with the `uv` package manager.
- `uv run python -m img_search.pipeline.embed` runs the embedding pipeline using the defaults in `embed_config.yaml`; override Hydra choices inline, e.g. `uv run python -m img_search.pipeline.embed models=siglip2 datasets=inquire logging=debug`.
- `uv run pre-commit run --all-files` executes the Ruff-based lint suite before you push.

## Coding Style & Naming Conventions
- Target Python 3.13, use 4-space indentation, and favour type hints for new APIs (see `img_search/proto/embed_result.py`).
- Module and file names stay snake_case; classes are PascalCase; configuration nodes mirror their Hydra groups.
- Formatting and linting ride on Ruff (`E,F,I,B,UP,A,C4`). Let it manage import ordering and common pitfalls instead of manual tweaks.

## Testing Guidelines
- Current smoke coverage lives in `tests/embedding/*.py`; invoke them with `uv run python tests/embedding/jina.py` or `.../siglip2.py` to validate encoder integrations.
- When adding tests, place them under `tests/` with descriptive snake_case filenames and structure them as executable scripts or (preferably) pytest-style functions. Add any new test utilities to `img_search/utils/` to keep re-useable helpers centralized.
- Capture novel fixtures (e.g. sample images) inside `tests/imgs/`, and document any external datasets referenced by Hydra configs.

## Commit & Pull Request Guidelines
- Follow the existing log: a single, descriptive sentence starting with a capitalized verb (`Refactor ImageDataset ...`, `Enhance dataset ...`). Keep subjects under ~72 characters and add wrapped body paragraphs when nuance is needed.
- Pull requests should summarize the change, list Hydra overrides or datasets touched, note manual test commands executed, and link related issues. Include screenshots only when the output materially changed (e.g., new embedding metrics).
