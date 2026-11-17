---
name: API gotchas
description: Traps in the HTTP API layer: FastAPI returns 400 (not 422) for a body that is not UTF-8, so an always-200 endpoint must parse its own body.
source: runs/2026-10-14-fastapi-server
---

# API gotchas

## 2026-10-14 FastAPI returns 400, not 422, for a body that is not UTF-8
- **Area:** api
- **Symptom:** with a 422 handler override, `POST /ask` with `{"question": "\xff\xfe"}` still returned 400 `{"detail": "There was an error parsing the body"}`.
- **Cause:** fastapi/routing.py turns JSONDecodeError into RequestValidationError (422) but any other parse exception into HTTPException(400).
- **Fix:** /ask reads the raw body and validates with `AskRequest.model_validate_json` in the handler's own try block.
- **Avoid next time:** for an "always 200" endpoint, own the parsing; don't rely on exception-handler overrides.
