---
name: Build decisions
description: Decisions about how the service is installed and started: ./serve syncs with --inexact so a dev checkout keeps its dev tools.
source: runs/2026-10-14-fastapi-server
---

# Build decisions

## 2026-10-14 ./serve syncs with --inexact
- **Area:** build
- **Chose:** `uv sync --frozen --no-dev --inexact`, then `exec uv run --frozen --no-dev python -m corpusqa.server`.
- **Over:** `uv sync --frozen --no-dev` (exact).
- **Why:** `uv sync` removes extraneous packages by default, so in a dev checkout ./serve would uninstall pytest/ruff/pyright from the shared .venv — including mid-run when tests/contract/test_serve_script.py starts ./serve under pytest. A fresh clone gets the same result either way. `uv run` is already inexact.
