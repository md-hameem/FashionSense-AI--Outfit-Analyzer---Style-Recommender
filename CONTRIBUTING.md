# Contributing to FashionSense AI

Thank you for your interest in contributing. This document covers branch strategy, commit conventions, and the PR workflow.

---

## Branch Strategy

```
main          — stable, tagged releases only
dev           — integration branch; all PRs merge here first
feature/*     — new features        e.g. feature/llm-suggestions
fix/*         — bug fixes           e.g. fix/gradcam-hook-cleanup
chore/*       — tooling, deps, CI   e.g. chore/bump-transformers
docs/*        — documentation only  e.g. docs/api-reference
```

**Rule:** Never push directly to `main`. All changes go through a PR into `dev`, then `dev` → `main` for a release.

---

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

**Types:** `feat` | `fix` | `chore` | `docs` | `test` | `refactor` | `perf` | `ci`

**Examples:**
```
feat(models): add ViT-B/16 fine-tuning with last-4-block strategy
fix(datasets): handle missing annotation JSON files in DeepFashion2
chore(deps): bump transformers to 4.41.0
docs(readme): add Kaggle notebook quickstart section
test(trainer): add unit tests for MixUpCrossEntropy with lam=0
```

---

## Pull Request Checklist

Before opening a PR:

- [ ] Branch is up-to-date with `dev`
- [ ] All existing tests pass: `pytest tests/ -v`
- [ ] New code has corresponding tests (target: 85% coverage)
- [ ] No secrets, credentials, or dataset files staged
- [ ] Model checkpoints excluded (use `.gitignore`)
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] PR description explains *why*, not just *what*

---

## Development Setup

```bash
git clone https://github.com/your-username/fashionsense-ai.git
cd fashionsense-ai
git checkout dev

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
cp .env.example .env          # fill in local values
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/test_datasets.py -v
```

---

## Reporting Issues

Open an issue with:
- A clear title describing the problem
- Steps to reproduce
- Expected vs actual behaviour
- Python version, OS, GPU (if ML-related)
- Relevant logs or error tracebacks
