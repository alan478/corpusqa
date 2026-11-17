---
name: API decisions
description: Decisions about the /ask and /health endpoints: /ask parses its own body (always 200), size caps by characters and id count, blank answers become the internal-error answer, /health says "failed" (still 503) when start-up raises; a cut of the id list (over MAX_DOCUMENT_IDS, kept at 500 by the user) is logged with how many were dropped and put on the ask trace.
source: runs/2026-10-14-fastapi-server, runs/2026-10-16-agent-tools
---

# API decisions

## 2026-10-16 A cut of the id list is logged and put on the ask trace (the 500 cap stays)
- **Area:** server
- **Chose** (user: "I agree with the 500 cap"; log and trace every cut): when more than
  `MAX_DOCUMENT_IDS` (500) known ids remain after dropping unknown ones and repeats,
  `respond._cited_ids` logs `cut N of M cited document ids: at most 500 are returned (the
  first ones, in the answerer's order)`, and the current observation (the `ask` trace) gets
  `document_ids_cited`, `document_ids_cut`, `document_ids_cap` and a WARNING. No cut: the
  trace is untouched. Through `obs.current()`, so `build_response`'s signature stays.
- **Over:** removing or lowering the cap (with `MAX_ANSWER_CHARS` it keeps a response
  provably under the contract's 1 MiB; an R# can hold 1,141 ids on data/); a new span per
  response.
- **Tests:** `test_a_cut_of_the_id_list_is_logged_and_recorded_on_the_trace` (507 known ids
  + one unknown: 7 cut, not 8), `test_no_cut_leaves_the_trace_alone`.

## 2026-10-14 Blank answerer output becomes the internal-error answer
- **Area:** api
- **Chose:** if the answerer returns an empty/whitespace answer, /ask returns `INTERNAL_ERROR_ANSWER` with `[]` (outcome `empty_answer`), and `build_response` also guards it.
- **Over:** a generic "no answer" text, or keeping the ids.
- **Why:** a blank answer is a pipeline bug, not evidence the corpus lacks the answer (AGENTS.md: internal error never looks like "not found"); ids without text support nothing.

## 2026-10-14 /health says "failed" (still 503) when start-up raises
- **Area:** api
- **Chose:** body `{"status": "failed"}` with 503 after a start-up exception; `{"status": "starting"}` while running; logged at CRITICAL with traceback.
- **Over:** exiting the process, or reusing "starting".
- **Why:** the grader reads only the code, so the distinct body only helps a human debugging; the process stays up so the log is easy to read.

## 2026-10-14 Size cap by characters and id count, not by measuring bytes
- **Area:** api
- **Chose:** `MAX_ANSWER_CHARS = 64_000` (+ truncation marker) and `MAX_DOCUMENT_IDS = 500`, both in config.py.
- **Over:** serialising and trimming to an exact byte budget.
- **Why:** one JSON string char is at most 6 bytes, so 64K chars ≤ 384 KiB; 500 ids ≈ 22 KiB: provably < 1 MiB with no loop. A unit test renders the worst case and checks < 1 MiB.

## 2026-10-14 /ask parses its own body instead of a typed FastAPI body param
- **Area:** api
- **Chose:** `ask(request: Request)` reads `await request.body()` and validates with `AskRequest.model_validate_json` inside one try block.
- **Over:** `ask(payload: AskRequest)` + overriding the 422 RequestValidationError handler.
- **Why:** FastAPI has a second rejection path: body parse errors that are not JSONDecodeError (e.g. invalid UTF-8) raise HTTPException(400) before the handler, which a 422 override does not catch. One try block covers every path; a test sends invalid UTF-8.
