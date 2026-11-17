---
name: Agent — the loop, its model client, tools, citations and compaction (as built)
description: The agent as built (agent/, llm/client.py GeminiChat, prompts/) - one Agent object per question with a prompt of [system, question, *history]; the code-side first search (the question searched as asked, before step 1); steps with a budget note; the stop rule in agent.md and the limits that force a finish (10 steps, 25 tool calls, a 30 s reserve, a failed call, two replies without a tool call); fallback and AgentError; the deadline chain from the server; compaction into a working memory (skipped, not replaced, when the call fails); the ToolBox (errors as text, ToolInputError messages) and finish; D#/R# citations resolved by code and handles in the answer text named by code; GeminiChat (raw turns re-sent, stand-in thought signature, limiter, ledger, retries, thinking level HIGH); settings; tracing; how the smoke, tune and held-out evals shaped it and why; known limits; tests. Read before changing the agent loop, its prompts, its limits, the chat client or the thinking setting. The tools themselves are corpus-tools.md.
source: runs/2026-10-16-agent-tools, runs/2026-10-16-corpus-tools
---

# Agent (as built)

One question gives one `Agent` object: a ReAct loop in which `gemini-3.8-flash` calls tools
until it calls `finish`. Code does what code can do exactly (search, filters, counts, id
lists, citations); the model decides what to look up and writes the answer. The shape
follows the user's assignment-1 harness (one class; a history of model turns and tool
messages linked by call id; every error as text; compaction of an old prefix), with one fix:
our compaction really replaces the old prefix, and a test checks that the prompt shrinks.

The tools it calls: `corpus-tools.md`. The HTTP side and the wiring: `http-server.md`.
Why each part is as it is: `../decisions/agent.md`, `../decisions/llm.md`,
`../decisions/prompts.md`. Runs and scores: `docs/eval-log.md`. Why each prompt changed:
`docs/prompt-log.md`.

## Parts
| File | Role |
|---|---|
| `agent/agent.py` | `Agent(question, model, tools, *, deadline, clock, limits, catalog, first_search)`: `build_prompt`, `first_search_step`, `step`, `maybe_compact`, `compact`, `run`. `AgentLimits` (defaults from `config.py`), `AgentResult` (answer, citations, not_found, stop, forced_by, steps, tool_calls, usage, compactions), `CompactionEvent`, `AgentError`, `system_text`, `question_text`, `budget_text`, `rough_tokens`, `transcript`. |
| `agent/tools.py` | `Tool[A]` (schema, Pydantic arguments model, `run`), `ToolBox` (`schemas`, `run`, `run_one`), `check_arguments`, `schema_mismatches`, `FinishArgs`, `finish_tool`, `read_finish`. |
| `agent/session.py` | `ToolSession` protocol (`toolbox`, `resolve_citations`, `clean_answer`), `ToolSessionFactory`, `FinishOnlySession` (tests). |
| `agent/answerer.py` | `Answerer` protocol (`answer(question, time_left_s=None)`), `AnswerResult`, `AgentAnswerer`, `UnavailableAnswerer` + `AnswererUnavailable`, `PlaceholderAnswerer` (tests). |
| `agent/corpus_tools.py`, `agent/handles.py` | The 7 corpus tools, `CorpusToolSession`, handles (`corpus-tools.md`). |
| `llm/interfaces.py` | SDK-free types: `Role`, `ToolChoice` (`"auto"`, `"none"` or a tuple of allowed names), `ToolCall`, `Message`, `ToolSchema`, `Usage`, `ModelReply`, `ChatModel`. |
| `llm/client.py` | `GeminiChat` (`complete`, `from_settings`, `thinking_sent`), `to_gemini`, `from_gemini`, `gemini_config`, `usage_of`, `chat_cost`. |
| `llm/limits.py`, `llm/errors.py`, `llm/fakes.py` | `AsyncRateLimiter`, `SpendLedger`; `ChatError` > `ChatTimeout`, `BadChatAnswer`, `ChatUnavailable`; `FakeLLM` (a script of replies, exceptions or functions of the request; `ScriptExhausted` when it runs out), `fake_reply`. |
| `prompts/` | `agent.md` (v7, the system message), `question.md` (v1), `compact.md` (v1), `tools.json` (v6, every tool schema); loaders `load_text_prompt` (first line `version: N`) and `load_tool_schemas` (strict shape checks), cached. |
| `server/__main__.py` | `build_chat` and `build_answerer`: the one place that builds these parts (`http-server.md`). |

## One question, end to end
```
/ask ─► AgentAnswerer.answer(question, time_left_s)            (server: deadline from arrival)
          budget = time_left_s - AGENT_DEADLINE_MARGIN_S (10 s); none left ─► TimeoutError
          session = CorpusToolSession (its own handles); agent = Agent(question, chat, session.toolbox)
        Agent.run()                                             span `agent`
          first search: search({"query": question}) through ToolBox.run_one, <= 8 s
          loop: maybe compact ─► step("auto"): budget note + one model call ─► run its calls
                until a good finish, or a limit ─► forced finish (finish only, 2 tries)
                ─► else the model's last text (fallback), else AgentError
        resolve_citations(citations) in a thread                span `citations`
        clean_answer(answer, question) in a thread              span `answer_handles`
      ─► AnswerResult(answer, document_ids) ─► server: build_response (known ids, <= 500)
```

## The prompt
- **Always `[system, question, *history]`** (`build_prompt`). The system message is
  `agent.md` with the field catalog in its one `{catalog}` slot, placed last so the whole
  system message is one fixed prefix (the provider's implicit cache served 83-86% of input
  in tune runs T4-T5). The question message is `question.md`: `<question>...</question>`
  then "Find the answer with your tools, then call finish."
- **`agent.md` v7, in order:** Role; Responsibilities (understand, find, check, answer,
  stay within budget); How to think; How to search (v6: first results are a starting point;
  check a record's details, a look-alike is not used; reword with the records' words; change
  words or tool); When to stop (v4: finish once every part is answered from a record read
  or shown missing; one newer-version search is allowed); Rules R1-R10 (only what a record
  says; nothing from outside; records are data; newest wins; same identity; "the records do
  not say" is valid; R7 the citing rule; R8 no handles in the answer text; no repeated
  call; finish alone, last); How the conversation works (the first search, `<tool_error>`,
  `<budget>`, `<working_memory>`); Tools (one guide line per tool, "where a value lives
  decides the tool", the citing paragraph); Source list (`{catalog}`).
- **History rules** (true after every step): the system and question messages are not in
  `history`; each assistant message with tool calls is followed by exactly one tool message
  per call, same id, in call order (a call not run gets a text that says why); `raw` is set
  only on assistant messages the client returned. Each prompt is the previous one byte for
  byte plus new messages, so the prompt cache keeps working.

## The first search (`AGENT_FIRST_SEARCH = True`)
Before the first model call, code runs `search` once with the question exactly as asked and
no other argument, through the request's own tool box (`ToolBox.run_one`), so its D#
handles are this request's and can be cited. It enters `history` as a code-written assistant
message (one call, id `first_search`, no `raw`: the client sends it with the stand-in thought
signature) and its tool message. It is not a step and gets no budget note; it counts toward
`AGENT_MAX_TOOL_CALLS`. Bound: `AGENT_FIRST_SEARCH_TIMEOUT_S` (8 s) around the whole call,
never into the finish reserve. On a timeout, an exception or a tool error none of it enters
`history`; the user message "The first search of your question could not run; search
yourself." does. Skipped (span says why) when off, with no `search` tool, a blank question,
or no time. Why: on tune question pub_0077 the question's own text ranks the right record
first, but the model searched in the question's quoted words and never found it; advice in
the prompt had not changed how it searches.

## Steps, budget notes and the stop
- **A normal step:** every schema, `tool_choice="auto"`, timeout `min(time left - 30 s
  reserve, 40 s)`. At most `AGENT_MAX_CALLS_PER_STEP` (4) calls of one reply run, all at
  once; the rest get "Not run: at most 4 tool calls run per reply...".
- **Budget note:** before every model call after the first, one user message
  `<budget>Step 6 of 10. 4 steps left after this one.</budget>`; a forced finish gets
  "Out of budget. Only finish may be called now: answer from the records you have read." It
  is kept in `history` once the call returns (a failed call keeps none).
- **`finish`:** a good `finish` ends the run; the other calls of that reply get "Not run:
  finish was called in the same reply". Bad `finish` arguments give `read_finish`'s error
  text and the loop goes on. A reply with no tool call gets the nudge "Call a tool, or call
  finish with your answer." once.
- **The stop rule is in the prompt** ("# When to stop"); code enforces only the limits.
- **Forced finish** (`tool_choice=("finish",)`, every schema still sent; any other call is
  not run) after `AGENT_MAX_STEPS` (10) steps, `AGENT_MAX_TOOL_CALLS` (25) tool calls,
  30 s or less left (`AGENT_FINISH_RESERVE_S`), a failed model call (any `LLMError`), or a
  second reply without a tool call (`forced_by`: step_limit, tool_call_limit, deadline,
  model_error, no_tool_call). Its call gets `min(time left, 40 s)`. Tried
  `AGENT_FORCED_FINISH_TRIES` (2) times; then the model's last text (`stop="fallback"`, no
  citations, so no ids); with no text at all, `AgentError`, which the server turns into its
  internal-error answer. So a run makes at most 12 step calls.

## Time
The server's deadline counts from the request's arrival (150 s). `AgentAnswerer` takes the
time left and ends the agent's deadline `AGENT_DEADLINE_MARGIN_S` (10 s) before it, for the
last step's tools, the citations and the response. No model call runs past the agent's
deadline; the last 30 s are kept for the forced finish. Tool calls have no limit of their
own (Codex #1, open).

## Compaction
- **Estimate** before each step: the input tokens the model reported for the last call plus
  the rough size (JSON characters / `TOKEN_CHARS`, `raw` left out) of what was added since;
  the rough size of the whole prompt before any count. A reported 0 is no count.
- **Trigger:** over `AGENT_COMPACT_AT_TOKENS` (80,000), with more than
  `AGENT_COMPACT_KEEP_STEPS` (1) complete steps, and a call still fits before the reserve.
- **The call:** everything before the last complete step (an earlier working memory, the
  first search, notes) is written as a plain-text transcript (step numbers go on; the first
  search is "before step 1"; budget notes left out; every result whole) and sent with the
  question to one call: system `compact.md` (headings PARTS, FACTS, TRIED, RULED OUT,
  DISAGREEMENTS, LEADS, NEXT; rules C1-C6), no tools, `tool_choice="none"`, thinking budget
  256, output cap 256 + 1,500. The old part of `history` is replaced in place by one user
  message `<working_memory>...</working_memory>`; the last step stays word for word. Not a
  step; its usage counts in the totals. A memory that does not shrink the prompt is flagged
  `no_gain`.
- **A failed or blank compaction call is skipped** (user: "Skip compaction instead"):
  `history` stays whole, the event says why, and it is tried again before the next step.
  The model can read the whole history (1M-token window), so a failure costs tokens, not
  facts.
- The largest document is ~8,200 tokens, so several whole reads fit under the limit. The
  live test forced one compaction (2,382 -> 1,994 tokens); smoke run 1 had none.

## The tool box and `finish`
- `ToolBox(tools)` refuses two tools with one name, and a tool whose schema and arguments
  model disagree (property names, required names, `additionalProperties: false` vs
  `extra="forbid"`): a drift fails at start, not at the first model call. It keeps no
  per-call state.
- `run(calls)` runs a reply's calls at the same time (each body in `asyncio.to_thread`) and
  returns one tool message per call in call order. Every problem comes back as text in
  `<tool_error>` tags, never as an exception: an unknown tool (with the tool list), bad
  arguments (every failing field, Pydantic's message only, never the input; nothing run), a
  `ToolInputError` (its message, written for the model), any other exception (only its type;
  the traceback goes to the log). Nothing is ever cut: each tool pages its own output.
- `finish(answer, citations, not_found)`: all three required; a blank answer is refused;
  it returns "Answer received.". The `finish` description (`tools.json` v6) says what to
  cite, matching R7.

## Citations and the answer text
- **What to cite** is the prompt's job: R7 (cite a record only if it states something the
  question asks for and the answer uses it; background, records read while looking, and a
  record the question names that lacks the asked fact are not citations; a whole-question
  "not found" cites nothing) and the citing paragraph (cite an R# for every record of a
  result set, else the D# used).
- **Resolving** is code's: `session.resolve_citations` (D# -> its id, R# -> every id of the
  set, known ids kept, anything else dropped, deduped). A fallback answer resolves nothing.
  The server keeps known ids only, at most 500.
- **Handles in the answer text** (R8 forbids them; the net catches the rest):
  `session.clean_answer` names each handle this request issued as its record's title,
  source and date, or an R# as "the N matching <source> records" (`corpus-tools.md`). If
  the net raises, the answer goes out as written.

## The model client (`GeminiChat`)
- `AGENT_MODEL = "gemini-3.8-flash"` (user approved after the step 1 probe: newest Flash,
  passed every harness check, 1-2 s per tool step); the SDK's async client, held for the
  instance's life.
- `to_gemini`: Gemini's own reply is re-sent as its `raw` Content, unchanged (its thought
  signatures are checked by the API); a model turn written by code gets
  `CHAT_STAND_IN_THOUGHT_SIGNATURE` on its first function call (without it: 400). User text
  messages in a row merge into one Content; tool results are their own Content; a function
  response carries its call's id only when the model turn before it has that id.
- `from_gemini`: a reply is usable only when it stopped with STOP, MAX_TOKENS or no reason
  (MAX_TOKENS logs a warning); anything else is `BadChatAnswer`, after the call is charged.
  An empty call id becomes `call_<n>`; an absent token count is 0.
- `complete`: the spend reservation first (`GEN_SPEND_CAP_USD` $40, ledger
  `.cache/spend/generation.json`; a call over the cap fails at once), then the rate limiter
  (`CHAT_REQUESTS_PER_MINUTE` 240), the call and up to 5 retries on 429 / 5xx (backoff 1 s
  doubling to 20 s), all inside the caller's timeout. A network error (`httpx.TransportError`,
  which the SDK raises as it is) becomes `ChatError` (`ChatTimeout` for a timeout) at once,
  not retried, so the agent's `except LLMError` recovery runs (Codex #5, fixed 2026-10-17;
  the embedder does the same with `EmbeddingError`). One `chat` generation per call.
- **Thinking:** `AGENT_THINKING_LEVEL = "high"`: every call that asks for
  `AGENT_THINKING_BUDGET` (each agent step, the forced finish included) is sent the level
  HIGH instead of the 1,024-token budget; compaction keeps its budget of 256. Thinking counts
  toward `max_output_tokens`, so the agent's cap is 1,024 + 16,000 (the longest answer the
  response can carry) = 17,024; the 40 s call timeout cuts a long think first.
- Prices `CHAT_PRICE_*` 0.75 / 0.075 / 3.75 USD per 1M tokens (input / cached / output); they
  double on 2027-01-01.

## Shared and per-request state
Shared, read-only or thread-safe: the `AgentAnswerer`, the `CorpusTools` factory,
`GeminiChat` (limiter and ledger), `SqliteReader` (a connection per call). Per question: a
new tool session (its handles) and a new `Agent`. At most 4 answers at once (`AskQueue`); the
tools run in threads, the chat calls are async.

## Tracing
One `agent` span per run (prompt versions of agent, question, compact and tools; the tool
names; limits; time left; catalog and system sizes; at the end stop, forced_by,
first_search, steps, tool calls, compactions and skipped ones, tokens, thinking, cached,
cost). Under it: `first_search` (call id, timeout, outcome ok / failed / timed_out /
skipped, reason, elapsed, result size, counted) with its `tool:search`; one `step` per model
call (number, tool choice, timeout, estimated tokens, budget note; output text and calls,
stop reason, input tokens, finished, calls not run) with its `chat` generation and
`tool:<name>` spans; one `compact` per compaction. Then `citations` and `answer_handles`
under `ask`. Details: `tracing.md`.

## How the evals shaped it, and why
All on the smoke (10) and tune (30) sets; held-out only at the user's milestones, as
aggregates, choosing nothing. Evidence and every number: `docs/eval-log.md`.
| Run | What changed | What it showed | Kept? |
|---|---|---|---|
| Smoke 1 | first run: `agent.md` v3, `tools.json` v4, thinking budget 1,024 | ids perfect, answers 6 correct / 4 partly; 3 of 10 used all 10 steps with every cited record read by step 3; 20 of 22 wasted calls were quoted searches that the keyword side OR-ed into single words (search never said "nothing found"); no stop rule; the model never saw its step count; 0 thinking tokens on 33 calls | - |
| Smoke 2 (L) | stop rule (`agent.md` v4, the user's completion policy + 3 tweaks) and the budget note | 9 / 1 / 0, no forced finish, cost -31% | yes |
| Smoke 3 (S) | exact phrases and keys in search (`SEARCH_EXACT_PHRASES`, `tools.json` v5) | same scores, honest "nothing found", cost -7% | yes |
| Smoke 4 (K) | thinking level HIGH | 10 / 0 / 0, but cost +52%, latency 3.6x | yes (keep-rule); user's call |
| Tune T1-T3 | HIGH vs LOW vs off on 30 questions | answers about equal (27 / 26 / 26 correct); every setting: unanswerable answers cite a background record (0-1 of 4 abstain), pub_0077 a false "not found" (vocabulary), the all-10-step questions are the ones with no answer, handles leak into answer text | - |
| (no run) | `agent.md` v5 (R7 citing rule, R8 no handles) + the code net; `tools.json` v6; `agent.md` v6 "# How to search" (anchoring bias); the first search in code + `agent.md` v7 | | measured in T4-T5 |
| Tune T4-T5 | v7 + the first search, HIGH vs LOW | HIGH 28 / 2 / 0, 116 of 118 facts, 4 of 4 abstain, F1 0.954, $0.039 a question, p50 32.8 s; LOW 25 / 5 / 0, 2 of 4 abstain, F1 0.967, $0.023, 9.8 s; pub_0077 fixed in both | HIGH chosen (before held-out) |
| Held-out | milestones v7-first-search-high / -low (49 questions, aggregates) | HIGH answerable F1 0.948, 7 of 7 abstain, p50 37 s, $0.046; LOW 0.926, 1 of 7: the tune gains hold with HIGH | confirms, changes nothing |

## Known limits (open; `docs/backlog.md`)
- Unanswerable questions still search until the step limit (about 25% of the spend); the
  step-10 note "No steps left after this one" is followed by a forced-finish step.
- The model often quotes `filter` argument names (`"\"field\""`); the call is refused and
  a step is lost.
- The citing rule is enforced by the prompt only (LOW broke it on held-out).
- Codex review findings #1 (tool time), #9 (failed calls traced thinly) and the Lows
  (#5, network errors, is fixed).
- A timed-out first search's thread is not stopped (Python cannot): it ends on its own.
- `config.py`'s comment on `AGENT_DEADLINE_MARGIN_S` still names the server's old
  `asyncio.wait_for` (it is `asyncio.timeout_at` now). A comment only; editing `config.py`
  rebuilds the index.

## Tests
`tests/unit/test_agent.py` (loop, limits, forced finish, budget notes, prompt pins),
`test_agent_compaction.py`, `test_agent_first_search.py`, `test_agent_answerer.py` (+ one
live end-to-end test with a forced compaction), `test_agent_tools.py`, `test_llm_chat.py`,
`test_llm_chat_client.py` (+ live tests), `test_prompts.py`, `test_gemini_chat_probe.py`
(live: the step 1 probe), `test_corpus_tools.py`; `tests/contract/test_server_agent.py`.
All offline tests use `FakeLLM`; live ones are marked `live` (`uv run tests-live`).
