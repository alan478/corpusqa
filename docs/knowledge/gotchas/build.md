---
name: Build gotchas
description: Traps with uv installs: `uv sync` is exact by default and strips dev tools from a shared .venv when run with --no-dev.
source: runs/2026-10-14-fastapi-server
---

# Build gotchas

## 2026-10-14 `uv sync` uninstalls dev tools when run with --no-dev
- **Area:** build
- **Symptom:** (avoided) ./serve's `uv sync --frozen --no-dev` would strip pytest/ruff from the shared .venv, also while the ./serve test runs under pytest.
- **Cause:** `uv sync` is exact by default; `uv run` is inexact.
- **Fix:** `uv sync --frozen --no-dev --inexact` in ./serve.
- **Avoid next time:** any script that syncs a shared venv with fewer groups needs `--inexact`.
