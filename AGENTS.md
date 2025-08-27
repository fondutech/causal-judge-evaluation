# Repository Guidelines

## Project Structure & Module Organization
- `cje/`: Core library (calibration, estimators, cfbits, data, diagnostics, utils, visualization, teacher_forcing, interface). Entry point via `python -m cje` is available.
- `cje/tests/`: Pytest test suite and fixtures.
- `docs/`: Sphinx docs; built HTML in `docs/_build/html/`.
- `examples/`, `experiments/`: Usage samples and reproducible studies.
- `Makefile`, `pyproject.toml`: Dev workflow and tooling configuration.

## Build, Test, and Development Commands
- Install: `make install` or `poetry install`
- Dev setup (install + pre-commit): `make dev-setup`
- Run tests: `make test` or `poetry run pytest cje/tests -v`
- Coverage: `poetry run pytest --cov=cje --cov-report=term-missing`
- Lint + type check: `make lint` (Black + mypy)
- Format: `make format`
- Docs: `make docs` and serve with `make docs-serve`

## Coding Style & Naming Conventions
- Language: Python 3.9–3.12 (poetry manages env).
- Formatting: Black (line length 88). Run `make format`.
- Types: mypy enforced; prefer explicit type hints and `disallow_untyped_defs` compliance.
- Naming: modules and functions `snake_case`, classes `CamelCase` (e.g., `CalibratedIPS`), constants `UPPER_SNAKE_CASE`.
- Docstrings: concise, argument/return types where not obvious.

## Testing Guidelines
- Framework: Pytest with fixtures in `cje/tests/`.
- Naming: files `test_*.py`, tests `test_*` functions; parametrize where helpful.
- Scope: add unit tests near touched logic and an integration test when behavior spans modules.
- Running locally: `poetry run pytest cje/tests -q` before opening a PR.

## Commit & Pull Request Guidelines
- Commit style: Conventional Commits (e.g., `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`). Keep messages imperative and scoped.
- PRs must include: clear description, motivation/approach, test coverage, and any doc updates. Link related issues.
- CI hygiene: ensure `make lint` and `make test` pass; attach benchmark notes for performance‑sensitive changes.

## Security & Configuration Tips
- Secrets: never commit API keys. Use `.env` (see `.env.example`) or `./set_secrets.sh`.
- Common env vars: `FIREWORKS_API_KEY`, plus optional vendor keys (OpenAI/Anthropic/Google) if used in experiments.
- Data paths: place sample data under `cje/tests/data/` or `examples/`, not in the repo root.
