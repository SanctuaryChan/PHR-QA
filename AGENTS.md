# Repository Guidelines

## Project Structure & Module Organization

- `PHR-QA/docs/`: design notes and references (start with `PHR-QA/docs/设计文档.md`).
- `PHR-QA/src/`: implementation code (currently a scaffold). The design doc proposes modules like `dataloader.py`, `graph_store.py`, `retriever_*.py`, `reader_llama.py`, and `eval.py`.
- Recommended (per design doc): `PHR-QA/configs/` (experiment YAMLs), `PHR-QA/data/` (datasets or symlink), and `PHR-QA/outputs/` (run artifacts).

## Build, Test, and Development Commands

This repository does not yet include standardized build/test scripts. When you add code, keep the workflow runnable from the repo root:

- Create an environment: `python -m venv .venv && source .venv/bin/activate`
- Install dependencies (if/when added): `python -m pip install -r requirements.txt`
- Run an experiment (intended entrypoint): `python PHR-QA/src/run.py --config PHR-QA/configs/webqsp.yaml`
- Run tests (once added): `pytest -q`

## Coding Style & Naming Conventions

- Python: 4-space indentation; `snake_case` for modules/functions; `CamelCase` for classes; type hints where practical.
- Keep modules small and single-purpose; prefer pure functions and explicit inputs/outputs over implicit globals.
- Put prompts and hyperparameters in config files (e.g., `PHR-QA/configs/*.yaml`) rather than hardcoding.

## Testing Guidelines

- Use `pytest`; place tests under `PHR-QA/tests/` and name files `test_*.py`.
- Prefer unit tests for graph traversal/scoring over end-to-end LLM runs; mock model calls and keep fixtures small.

## Commit & Pull Request Guidelines

- Git history is not available in this workspace; default to Conventional Commits (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
- PRs should include: what changed, how to run/verify, and example output paths (e.g., `PHR-QA/outputs/...`).

## Security & Configuration Tips

- Do not commit datasets, model weights, API keys, or caches. Keep large artifacts under `PHR-QA/data/` and `PHR-QA/outputs/` and add them to `.gitignore` when git is initialized.

## Agent-Specific Instructions

- Follow the offline evaluation pipeline described in `PHR-QA/docs/设计文档.md` (Data -> Planner -> Retriever -> Reader -> Eval -> Report).
- Phase 1 should operate on each sample's provided `subgraph` only (avoid full-graph DB integration until the pipeline is stable).
