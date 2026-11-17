---
name: Server gotchas
description: Traps when running the uvicorn server: after a graceful SIGTERM shutdown the exit code is 143, not 0; and a port already bound on 127.0.0.1 by another process hides our server (bound on 0.0.0.0) from local probes.
source: runs/2026-10-14-fastapi-server, runs/2026-10-14-ingestion
---

# Server gotchas

## 2026-10-16 A port already bound on 127.0.0.1 hides our server
- **Area:** server / dev (measuring start-up)
- **Symptom:** a start-up probe on port 8765 polled /health for 10 minutes and got 404, although the log said "/health is now 200".
- **Cause:** another local process listened on 127.0.0.1:8765; ours binds 0.0.0.0:8765 (macOS allows both), and requests to 127.0.0.1 reached the other one.
- **Fix:** `lsof -nP -iTCP:<port>` before choosing a port; reran on a free port.
- **Avoid next time:** check the port first, and treat any /health answer that is neither 503 nor 200 as "not our server".

## 2026-10-14 uvicorn exits with the signal, not 0, after a graceful SIGTERM shutdown
- **Area:** server
- **Symptom:** `./serve` exit code 143 after SIGTERM, although the log shows "Application shutdown complete".
- **Cause:** uvicorn's `capture_signals` re-raises captured signals after shutdown (Unix convention); uv then exits 128+15.
- **Fix:** none needed (CONTRACT only needs a stop within 10 s); tests assert the shutdown log line, the exit time and an empty process group, not the code.
- **Avoid next time:** do not assert exit code 0 for a SIGTERM'd uvicorn.
