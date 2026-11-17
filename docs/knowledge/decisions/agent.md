---
name: Agent decisions
description: Decisions about the agent loop and its tools - the Codex review triage of the harness and corpus tools (fixes wait for the user; #7 rejected); the code-side first search (agent.md v7); the citing rule plus a code net that names leaked handles; keeping the stop rule and step count (L); one change at a time after the first smoke eval; a failed compaction is skipped, not replaced; per-question tool sessions and the AgentAnswerer deadline; the token estimate and compaction call; how the loop reads the plan (forced finish after 10 steps, retries, fallback); how much each tool returns (pages, whole documents, error middle way); no cut on tool results; how finish is read; tool error texts and spans; a generic Tool checked at construction; the harness shaped like assignment-1; handles instead of raw ids; the field catalog on the tool factory.
source: runs/2026-10-16-corpus-tools, runs/2026-10-16-agent-tools
---

# Agent decisions

## 2026-10-17 Codex review of the agent harness and corpus tools: triage, fixes wait for the user
- **Area:** review
- **Review:** `docs/reviews/2026-10-17-agent-harness-and-tools.md` (gpt-6-astra, read-only,
  diff `21000f7c..cdc765eb`, committed in bca61006): 2 High, 7 Medium, 1 Low; the triage at
  the end of that file checked each finding against the code.
- **Chose:** accept the findings as triaged, but fix none yet: other agents were changing
  this code during the review, and the user decides which fixes run and when (proposed
  order: #5 and #1, then #4, then #9; then a smoke and a tune run). After triage:
  - accepted, open: #1 [High] tool time not bounded by the deadline; #5 [Medium] network
    errors skip the agent's recovery (wider than stated: every step and the search's query
    embedding); #4 [Medium] `contains` sees only 256 characters and pulls in other
    spellings; #9 [Medium] failed model calls traced thinly; Lows #2 (body read outside the
    deadline: Codex said High), #3 (ledger written on the event loop), #6 (filled-in call
    ids can collide), #8 (a JSON null shown as "absent"), #10 (no reason per citation on the
    span); plus, not Codex's, very large `bm25` / `dense` spans.
  - rejected: #7 (search pages without an exact total): the user's own rule of 2026-10-16
    ("Search deeper"); the page always says that more follow, and where.
- **Why:** each open finding is in `docs/backlog.md` item 6 with its fix, so the closed task
  loses nothing. None changed an eval result (0 errors and 0 timeouts in 288 eval requests).

## 2026-10-17 The first search: code searches the question before step 1 (agent.md v7)
- **Area:** agent / prompts
- **Chose** (user: "build the search as well first so we send the search result for the
  first question as well"): before the first model call, `Agent.first_search_step` runs
  `search` once with `{"query": question}` through the request's own tool box
  (`ToolBox.run_one`, new: the message plus the error kind), so its D# handles are this
  session's. It enters the history as a code-written assistant message (one call, id
  `first_search`, no `raw`, so the stand-in thought signature) and its tool message. Not a
  step: step numbers and budget notes do not move; it counts toward `AGENT_MAX_TOOL_CALLS`
  when its result is in the history. Bound `AGENT_FIRST_SEARCH_TIMEOUT_S` (8 s, `asyncio.wait_for`)
  around the whole call, never past the finish reserve; on a timeout, an exception or a tool
  error nothing of it enters the history, and "The first search of your question could not
  run; search yourself." does. Skipped (with the reason on the span) when off, with no
  `search` tool, a blank question, or no time. Compaction folds it in but never counts it as
  a step ("before step 1"). `agent.md` v7: one line under "# How the conversation works".
- **Over:** a prompt rule "search the question in plain words first" (the model does not
  follow query advice: 16 of 25 searches on pub_0077 used OR although `tools.json` said it
  is not needed, and it costs a turn); not counting it toward the 25-call cap; key lookups on
  key-like tokens of the question too (one change at a time); bounding only the embedding.
- **Why:** pub_0077 (`eval/results/tune/ask/20261017T034320Z/pub_0077_investigation.md`):
  the question's own text ranks the right record first, but the model searched in the
  question's quoted words and never found it.
- **Result:** tune T4-T5 (`docs/eval-log.md`): the code-written call id was accepted by the
  API; the first search ran on all 60 questions (~0.5 s); pub_0077 fixed in both settings;
  18 of 26 answerable questions in T5 cite only records from it.
- **Known limits:** a timed-out search's thread is not stopped (Python cannot); it ends on
  its own and may still give out handles the model never sees. With the switch off, v7's
  line is not true: compare "off" against v6.

## 2026-10-17 The citing rule (agent.md v5) and a code net that names leaked handles
- **Area:** prompts / agent (answer text)
- **Chose** (user: "fix the handle leak in two layers, and add the citing rule to the
  prompt"):
  1. `agent.md` v5 (bf224a40): R7 - cite a record only if it states something the question
     asks for and the answer uses it; background, records read while looking, and the record
     the question names when it lacks the asked fact are not citations; a whole-question
     "not found" cites nothing; a partial answer cites only the found parts. New R8 - never
     write D#/R# in the answer text; name the record by title, source and date. Step 4 of
     the responsibilities aligned with R7. Old R8/R9 renumbered (nothing refers to numbers).
  2. Code net: `ToolSession.clean_answer(answer, question) -> CleanedAnswer`, called by
     `AgentAnswerer` for every answer (fallback text too) in an `answer_handles` span.
     `CorpusToolSession` uses `Handles.name_in_text`: only handles this run issued, in their
     issued spelling, as whole tokens; a D# becomes `"<title>" (<source>, <YYYY-MM-DD>)`
     (inside a bracket `"<title>", <source>, <date>`), an R# `the 14 matching hubspot
     records`. `document_ids` are not touched. If the net raises, the answer goes out as
     written (logged, span ERROR).
  3. Then `tools.json` v6 (598b543d): the `finish` `citations` text no longer says "Include
     every document a fact in the answer came from", which pulled toward citing background.
- **Over:** changing every D#/R#-shaped token (the corpus can hold real ones, e.g. "D7
  retention"; tokens the question holds name something real); deleting handles (the
  sentence would lose what it pointed to); the prompt rule alone (a rule is followed most of
  the time, not always).
- **Why:** tune T1-T3: every one of the 4 unanswerable answers said the records lack the
  answer, then cited one background record (0 of 4 abstained); T3 wrote "(documented in email
  thread D3)" and "(D4)".
- **Result:** T4 (HIGH): 4 of 4 unanswerable questions cite nothing; held-out HIGH 7 of 7.
  LOW follows R7 less (T5 2 of 4, held-out 1 of 7); nothing in code checks it (backlog).

## 2026-10-17 Keep L (stop rule + step count, agent.md v4) after smoke run 2
- **Area:** prompts / agent (loop)
- **Chose:** keep L: the `# When to stop` section (prompts.md, v4 entry) and the budget note
  before every model call after the first: `<budget>Step N of 10. K steps left after this
  one.</budget>`, a new user message, so the prompt prefix stays byte-identical (cache kept).
  Smoke run 2 against run 1: ids the same (perfect), abstain the same, answers 9 / 1 / 0
  (6 / 4 / 0), facts 45 of 46 right (43), forced finishes 0 (3), model calls 50 (68), p50 /
  p95 8.4 / 18.3 s (15.6 / 35.4), $0.410 ($0.596).
- **Over:** reverting (nothing the keep-rule watches dropped); bumping `agent.md` to v5 for
  the final stop-rule text (the earlier v4 draft never ran in an eval).
- **Why:** AGENTS.md keep-rule. Still open after it: pub_0028 used all 10 steps (change S
  next), and thinking fell to 2 of 10 questions (change K).

## 2026-10-17 Three changes from the first smoke eval, one at a time
- **Area:** retrieval (search) / agent (loop, prompt) / llm (thinking)
- **Chose:** (1) exact matching for quoted text and keys in search (S); (2) a stop rule in
  `agent.md` and the step count shown each step (L); (3) the agent steps' thinking asked for
  by level (HIGH) instead of only a budget (K). One change at a time, each measured on smoke
  against the run before and kept only if it does not regress.
- **Over:** raising `AGENT_MAX_STEPS` (the long runs had everything by step 3); several
  changes at once (no way to tell which helped).
- **Why:** run 1 (`eval/results/smoke/ask/20261017T013336Z/`): 3 of 10 questions used all
  10 steps although every cited record was read by step 3; 20 of 22 wasted calls were quoted
  searches that the keyword side OR-ed into single words, so no search ever came back empty;
  no stop rule, no visible step count; 0 thinking tokens on all 33 calls of those runs.
  User: "record these findings for the first eval, we will report it; then also add why we
  are updating the search from our initial test, same thing for loop and same thing for
  thinking".

## 2026-10-16 A failed compaction is skipped, not replaced by code's short memory
- **Area:** agent
- **Chose** (user: "Skip compaction instead"): when the compaction call fails (an
  `LLMError`) or writes only blanks, `compact()` replaces nothing: `history` stays whole. It
  returns a `CompactionEvent` with `path="skipped"`, `steps_replaced=0` and a `reason`; the
  `compact` span gets a WARNING; the same line is logged. It is tried again before the next
  step (the prompt is still over the limit). The `agent` span splits `compactions` and
  `compactions_skipped`. Gone: `fallback_memory` and `COMPACT_FALLBACK_PREVIEW_CHARS`. Near
  the deadline the compaction call's timeout is `min(time_left - reserve, call_timeout)`, so
  a failure ends as the reserve starts and the forced finish keeps all of it.
- **Also fixed** (found by the new tests): `_limit_reached` now forces at `time_left <=
  reserve` (was `<`): at exactly the reserve a normal call would get 0 s
  (gotchas/agent.md).
- **Tests changed to the new rule** (the user replaced the rule they encoded):
  `test_a_failed_compaction_keeps_the_history_whole`,
  `test_a_skipped_compaction_keeps_an_earlier_working_memory`,
  `test_a_skipped_compaction_is_a_warning_on_its_span`; new: tried again at the next step,
  near the deadline never blocks the forced finish, `test_exactly_the_reserve_left_forces_a_finish`.
- **Over:** code's one-line-per-step memory with 120-character previews (it dropped most of
  each result, against the user's rule that nothing is dropped without the model being able
  to reach it); a cap on retries.
- **Why:** the model can read the whole history (1M-token window): a failed compaction costs
  tokens, not facts.

## 2026-10-16 Tool sessions and the AgentAnswerer
- **Area:** agent
- **Chose:**
  - `agent/session.py`: `ToolSession` protocol (`toolbox`, `resolve_citations(citations) ->
    list[str]`; later `clean_answer`), `ToolSessionFactory = Callable[[], ToolSession]`,
    `FinishOnlySession`. The corpus tools' `CorpusToolSession` matches it structurally
    (pyright checks it).
  - `AgentAnswerer(model, session_factory, *, ask_deadline_s, deadline_margin_s, limits,
    clock, catalog, first_search)`: per question a new session and a new `Agent`; deadline =
    start + (time left, or `ask_deadline_s`) - `deadline_margin_s`. A margin below 0 or not
    below the deadline is refused.
  - `AGENT_DEADLINE_MARGIN_S = 10.0`: every model call already ends by the agent's deadline;
    the margin covers the last step's tools, the citations and the response (140 s for the
    agent: more than the 30 s reserve + one 40 s call).
  - `document_ids = session.resolve_citations(result.citations)` in a thread (a session may
    read SQLite), in a `citations` span; `[]` after a fallback; `AgentError` propagates.
- **Over:** resolving in the event loop; giving a fallback's citations (none were checked);
  exporting the new names from `agent/__init__.py` (a merge conflict with the tools track).
- **Live test** (`test_live_end_to_end_with_a_real_compaction`): real `GeminiChat`, a
  test-only glossary tool, `compact_at_tokens=1`: stop `finish`, 3 steps, 3 tool calls, one
  compaction by the model (2,382 -> 1,994 tokens), both ids resolved, $0.00559.

## 2026-10-16 Token estimate and the compaction call
- **Area:** agent
- **Superseded in part by:** agent.md#2026-10-16 A failed compaction is skipped (the
  code-written fallback memory below is gone).
- **Chose:**
  - Estimate as plan 6.4 (last reported input tokens + rough size added since), plus: a
    reported 0 is "no count" (Gemini leaves absent counts out); after a compaction the base
    moves down by the rough size removed, else the estimate stays over the limit until the
    next call. `no_gain` compares the rough size of the whole prompt before and after.
  - The call: system `compact.md`, one user message `<question>...</question>\n\n<transcript>
    ...</transcript>`, no tools, `tool_choice="none"`, thinking budget 256, output cap 256 +
    1,500 (thinking counts toward the cap). Timeout as a normal call's. Its usage counts in
    the totals; it is not a step.
  - The transcript: one block per old message (`[step N: the agent]` with its calls,
    `[step N: result of name, call id]` with the result whole, `[earlier working memory]`,
    `[note to the agent]`); step numbers go on across compactions.
  - Compaction runs before every step, forced finishes included, only when a call still
    fits before the reserve; the limits are checked again after it. The old part is replaced
    in place (`history[:cut] = [memory]`); a memory is recognised by its tags.
  - (Superseded) a code fallback memory, one line per old step with 120-character previews,
    when the call failed or was blank.
- **Over:** `max_output_tokens=COMPACT_MAX_OUTPUT_TOKENS` alone (thinking would eat it); a
  thinking budget of 0 (not probed on this model).

## 2026-10-16 How the loop reads the plan (limits, forced finish, failures)
- **Area:** agent
- **Chose:**
  - `AGENT_MAX_STEPS` counts the steps that may call any tool; after that many the next step
    is a forced finish, and the forced finish and its one retry come on top (at most 12 step
    calls).
  - A forced step sends every schema with `tool_choice=("finish",)` (Gemini: ANY with
    `allowed_function_names`); any other call in it is not run ("only finish may be called
    now").
  - A good `finish` runs through the ToolBox like any call ("Answer received."); the other
    calls of that reply get a "not run" text, so the history rules hold. A bad `finish` runs
    through the box, which gives `read_finish`'s exact error text.
  - A model call that fails with an `LLMError` in a normal step forces the finish
    (`model_error`). Any other exception is a bug and propagates.
  - Time: a normal call gets `min(time_left - reserve, AGENT_CALL_TIMEOUT_S)`; a forced one
    `min(time_left, AGENT_CALL_TIMEOUT_S)`; with no time left no forced attempt is made.
  - The nudge and the "not run" texts are code constants in `agent/agent.py`, in the same
    `<tool_error>` tags as the ToolBox's errors. Limits are one `AgentLimits` dataclass.
    `AGENT_FORCED_FINISH_TRIES = 2`. `AgentResult.forced_by` says why. Each `chat` generation
    records `agent.md`'s version; the `agent` span all prompt versions.
- **Over:** the forced finish inside the 10 steps; sending only the `finish` schema in a
  forced step (the history then holds calls to undeclared tools; not probed); ending the run
  on a failed normal step.
- **Why:** every stop path gives an answer or `AgentError`, and no call the model asked for
  disappears without a tool message.

## 2026-10-16 How much each tool returns (the user's four answers)
- **Area:** agent / retrieval (tools)
- **Chose** (user, multiple-choice answers): (1) filter: exact count always; rows 50 at a
  time with "rows 1-50 of 278; next: offset=50" ("Pages, nothing lost"); (2) search: top 8
  documents with snippets, the next 8 with `offset=8`; (3) read: always the whole document
  (the longest is 32,959 characters, ~8,200 tokens); (4) tool errors: the "middle way" - a
  tool's own input errors pass their message, written for the model; any other exception
  shows only its type.
- **Over:** filter "ask to narrow" or "all rows at once"; search without paging or a fixed
  25; read in parts; error type only.
- **Why:** the user's rule: nothing may be dropped without the model being told and able to
  reach it. Any other limit on what a tool returns goes to the user first.

## 2026-10-16 No cut on tool results: every result goes to the model whole
- **Area:** agent
- **Chose:** no size cut in the ToolBox (no `cut_result`, `TOOL_RESULT_MAX_CHARS`,
  `TOOL_RESULT_MIN_CHARS`, `cut` span field; the span keeps `result_chars`). Each tool
  controls its own output size by paging.
- **Over:** the first build's cut at 10,000 characters (start + elision line + end).
- **Why:** user: "we already have guards right if the tool fails throw an error and just
  sent it message, remove the truncation". Compaction shrinks long history.

## 2026-10-16 How the agent reads `finish`
- **Area:** agent
- **Chose:** `read_finish(call) -> FinishArgs | str`: the checked arguments, or the exact
  error text `ToolBox.run` sends for that call (one shared `check_arguments`); `ValueError`
  for a call that is not `finish`. `FinishArgs` is frozen, `extra="forbid"`; a blank answer
  fails with "must not be blank"; the answer is kept as given. `FINISH_RESULT` = "Answer
  received." (the `finish` description promises that text: change both together).
- **Over:** a second validation path in the agent (the two could drift).

## 2026-10-16 Tool error texts and the tool spans
- **Area:** agent / tracing
- **Chose:** texts (constants in `agent/tools.py`): `<tool_error>No tool named 'x'. Tools: a,
  b, c.</tool_error>`; `<tool_error>Bad arguments for 'x': <loc>: <msg>; ... Nothing was
  run.</tool_error>` (every failing field, Pydantic's message only, never the input value);
  `<tool_error>'x' failed: <ExceptionType>.</tool_error>`; a `ToolInputError`'s message
  (added with the corpus tools). A tool that returns something other than text counts as
  failed. Span `tool:<name>`: input `given`, then `given` + `checked`; output the text as the
  model saw it; metadata `tool_call_id`, `error` (`unknown_tool` | `bad_arguments` |
  `tool_input` | `tool_failed` | null), `result_chars`. ERROR for a tool that raised;
  WARNING for the model's own mistakes. (The first build also cut long results: removed the
  same day, entry above.)
- **Over:** exception text in the model's text (it may hold paths, values or keys); ERROR
  for model mistakes (expected, and the run goes on); keeping the texts in `prompts/`
  (they are code-made templates).

## 2026-10-16 `Tool` is generic, and the box checks each tool at construction
- **Area:** agent
- **Chose:** `Tool[A: BaseModel]` with `args: type[A]` and `run: Callable[[A], str]`;
  `ToolBox(Sequence[Tool[Any]])` refuses two tools with one name and a tool whose schema and
  arguments model disagree (`schema_mismatches`: property names, required names,
  `additionalProperties: false` vs `extra="forbid"`; types and descriptions not compared).
  The box keeps no per-call state.
- **Over:** `run: Callable[[BaseModel], str]` (a tool of one model would not type-check); a
  schema test per tool only.
- **Why:** pyright passes without casts; a schema that drifts fails at start with the tool's
  name, not as a confusing error at the first model call.

## 2026-10-16 The harness is shaped like assignment-1: one class, google-genai, harness first
- **Area:** agent / llm
- **Chose:** model the harness on the user's assignment-1 (CMU 11-768) harness: prompt =
  system + question + `history` of assistant and `tool` messages linked by `tool_call_id`;
  one loop; tool schemas in their own file; every error as text; token estimate = last
  reported input tokens + rough size added since; compaction of an old prefix keeping the
  latest complete step. One `Agent` class (user: "you don't have to create a general class,
  just one is enough"). Messages are plain dataclasses; one function (`to_gemini`) converts
  them for google-genai and re-sends Gemini's own replies unchanged (user picked option a).
  Built the harness first, the real tools after (user: "the type of tools depends on what
  the other model does ... lets just build the harness around the model").
- **Over:** separate `Conversation` / `ToolRegistry` / loop modules; a base class with
  subclasses; the `openai` library against Gemini's OpenAI-compatible endpoint (a second
  library for the same API).
- **Not copied:** assignment-1's `compact_context` never writes the summary back into
  `history` and always summarises `history[2:20]` (its saved trajectory: 18 compactions, 0
  that made the prompt smaller). Ours replaces the prefix, and a test checks it shrinks.

## 2026-10-16 Handles instead of raw ids in tool texts
- **Area:** tools
- **Chose:** results show D# per document and R# per filter result or key lookup; raw
  dataset_doc_uuids are never printed (a known raw id is still accepted and cited).
- **Over:** printing ids (37 characters each).
- **Why:** shorter texts; the model cites handles and the session expands them exactly.

## 2026-10-16 The field catalog is a property of the factory
- **Area:** tools, prompts
- **Chose:** `CorpusTools.catalog` (text: per source its size, date range and common fields
  with kind, presence, distinct values, top 5 values, "same as" groups and "present when"
  conditions; fields mostly noise left out), built once at start-up.
- **Over:** building it in `agent.md` / `Agent` (the other track's files at the time).
- **Why:** ownership; on data/ it is 132 fields, 21,650 characters. It went into the system
  prompt with `agent.md` v3 (prompts.md, "agent.md v3").
