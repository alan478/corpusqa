---
name: LLM (chat model) decisions
description: Decisions about the agent's chat model and its client - thinking level HIGH chosen from the tune set before any held-out run (after smoke run 4 kept it by the keep-rule at +52% cost); gemini-3.8-flash and a $40 generation spend cap (user approved after the live probe); the probe's findings (thought signatures, several calls per reply, thinking counts toward max_output_tokens, prices); GeminiChat's conversion and call rules; output cap, thinking budget, pacing and retry values. Embedding decisions are in embeddings.md.
source: runs/2026-10-16-agent-tools
---

# LLM (chat model) decisions

## 2026-10-17 Thinking HIGH chosen from the tune set, before any held-out run
- **Area:** llm / eval
- **Chose:** `AGENT_THINKING_LEVEL = "high"` (as committed) as the shipped setting, from tune
  runs T4-T5 (`docs/eval-log.md`): HIGH 28 / 2 / 0 correct, 116 of 118 facts, 4 of 4
  unanswerable abstains; LOW 25 / 5 / 0, 113 of 118, 2 of 4. Recorded BEFORE the held-out
  milestone runs `v7-first-search-high` and `v7-first-search-low`, which the user asked for
  ("run the high and low as well on the rest of the question"). Those results are reported
  as aggregates only and choose nothing (AGENTS.md eval split); they confirmed the choice
  (HIGH answerable F1 0.948 and 7 of 7 abstains; LOW 0.926 and 1 of 7).
- **Over:** choosing after seeing both held-out runs (tuning on the test set); off or LOW
  (cheaper and faster: LOW $0.023 a question and p50 9.8 s against HIGH's $0.039 and 32.8 s,
  every HIGH question under 75 s of the 150 s deadline).
- **Why:** answers and citations are what is graded; HIGH leads on both on tune. This
  replaced the T1-T3 recommendation (off), made before `agent.md` v7 and the first search,
  when the three settings answered equally.

## 2026-10-17 Keep K (thinking level HIGH) after smoke run 4, by the keep-rule
- **Area:** llm (thinking) / agent
- **Chose:** keep `AGENT_THINKING_LEVEL = "high"` (`df4b2ea7`): in `GeminiChat`, a call that
  asks for `AGENT_THINKING_BUDGET` (every agent step, the forced finish included) is sent the
  level instead of the budget; compaction keeps its budget of 256; the generation records
  what was sent. Smoke run 4 against run 3: ids and abstain the same; answers 10 / 0 / 0
  (9 / 1 / 0); facts 46 of 46 (45); every call thought.
- **Over:** reverting to None. But K made worse what it was meant to improve: model calls 69
  (43), p50 27.7 s (7.7), p95 49.0 s (20.0), cost $0.583 ($0.383, +52%); pub_0028 still ran to
  step 10. Flagged for the user; settled by the tune runs (entry above).
- **Why:** the keep-rule (citations, abstain and the hand-checked answers must not get worse)
  held. Run 1 had 0 thinking tokens on all 33 calls of its long runs with the 1,024 budget;
  the probe showed a small prompt thinks ~358 tokens with HIGH against ~145 with the budget.

## 2026-10-16 Agent model and generation spend cap approved
- **Area:** llm / agent
- **Chose:** `AGENT_MODEL = "gemini-3.8-flash"` (the probe's recommendation) and
  `GEN_SPEND_CAP_USD = 40` in its own ledger `.cache/spend/generation.json`.
- **Over:** the Pro class (`gemini-3.1-pro-preview`, `gemini-4-argon`): slower and dearer;
  the fallback, one constant away (the answer judge uses `gemini-3.1-pro-preview`).
- **Why:** user, after reading the probe's findings: "okay move to 3". It is the newest
  Flash model, passed every check the harness relies on, takes 1-2 s per tool step (room for
  10 steps inside 150 s with 4 requests at once) and is 2.7x cheaper on input than the Pro
  class. Its price doubles on 2027-01-01 (comment in `config.py`).

## 2026-10-16 Live probe findings (step 1)
- **Area:** llm / agent
- **Probe:** `tests/unit/test_gemini_chat_probe.py` (marked `live`), google-genai 2.31.0,
  async client; 3 runs, 34 calls, $0.0303. Prompts tiny and invented.
- **Findings on `gemini-3.8-flash`:** several function calls in one reply work; the API gives
  call ids (`call_1383909`), so our `call_<n>` fill is a fallback only; ANY with
  `allowed_function_names=["finish"]` works and the `tools.json` schema
  (`additionalProperties: false`, via `parameters_json_schema`) is accepted; re-sending the
  model's Content unchanged, then one user Content with one `function_response` per call,
  works; two user Contents in a row are accepted. **A model turn rebuilt from our own calls
  without thought signatures gets 400 INVALID_ARGUMENT**; with the documented stand-in
  signature `skip_thought_signature_validator` on its first function call it is accepted
  (gotchas/llm.md). Thought signatures come on the first function-call part of a reply.
  `thinking_budget` and `thinking_level` (LOW, HIGH) are accepted; the model thinks only
  when needed. **Thinking counts toward `max_output_tokens`** (128 with budget 1,024: 118
  thinking + 6 answer tokens, MAX_TOKENS). `thoughts_token_count` is absent, not 0, when the
  model did not think. Latency per call 1.1-2.2 s (Pro class 2.6-5.0 s).
- **Prices** (USD per 1M, paid Standard tier, read 2026-10-16 from
  ai.google.dev/gemini-api/docs/pricing): `gemini-3.8-flash` 0.75 / 0.075 / 3.75 through
  2026-12-31, then double; `gemini-3.1-pro-preview` 2.00 / 0.20 / 12.00 (prompts <= 200k).
- **SDK:** google-genai does not retry unless `retry_options` is set, so our own retries do
  not stack on the SDK's. Not measured: the key's requests-per-minute quota.

## 2026-10-16 GeminiChat conversion and call rules
- **Area:** llm
- **Chose:**
  - `to_gemini`: only `user` text messages in a row merge into one Content; tool results are
    their own user Content; a user message after them is a separate Content; an empty
    assistant message is left out. A model turn code wrote gets
    `CHAT_STAND_IN_THOUGHT_SIGNATURE` on its first function call. A function response carries
    its call's id only when the model turn before it has a call with that id.
  - `from_gemini`: usable only when the reply stopped with STOP, MAX_TOKENS or no reason;
    anything else (SAFETY, MALFORMED_FUNCTION_CALL, ...) is `BadChatAnswer`, after the call is
    charged. An absent token count is 0.
  - `complete`: the spend reservation first (a call over the cap fails at once), then the
    limiter, the call and every retry inside `asyncio.timeout(timeout_s)`. The reservation is
    the request's JSON size / `TOKEN_CHARS` at the input price plus `max_output_tokens` at the
    output price. Only `APIError` is caught: 429 and 5xx are retried, others become
    `ChatError`; other exceptions are not wrapped (Codex #5, 2026-10-17: transport errors then
    skip the agent's recovery; open, backlog item 6).
  - The `chat` generation's input is the contents and config exactly as sent; a MAX_TOKENS
    reply logs a warning and marks the generation WARNING.
- **Over:** merging the nudge into the tool-results Content (not probed); the limiter before
  the reservation (it would wait for a slot just to be refused).

## 2026-10-16 Output cap, thinking budget, pacing and retry values
- **Area:** llm / config
- **Chose:** `AGENT_THINKING_BUDGET = 1_024`; `AGENT_ANSWER_MAX_TOKENS = MAX_ANSWER_CHARS //
  TOKEN_CHARS` (16,000); `AGENT_MAX_OUTPUT_TOKENS` = their sum (17,024).
  `CHAT_REQUESTS_PER_MINUTE = 240`, `CHAT_MAX_RETRIES = 5`, backoff 1 s doubling up to 20 s,
  `CHAT_HTTP_TIMEOUT_MS = 120_000` as a backstop only.
- **Over:** a small output cap (an answer that lists many matches would be cut); a larger
  budget (each 1,024 thinking tokens is ~7 s at the probe's speed).
- **Why:** thinking counts toward the cap, so the cap is the budget plus the longest answer
  the response can carry. The rate sits above the peak of 4 requests at ~1.5 s per call. The
  backoff cap is lower than the embedder's 60 s because every retry must fit inside the
  caller's timeout. (With the level HIGH the budget is not sent on agent steps; the 40 s call
  timeout then bounds a long think before the cap does.)
