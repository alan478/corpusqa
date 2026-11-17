---
name: add-agent-tool
description: Add or change a tool the ReAct agent can call (schema, code, observation, tests, tracing, eval). Use whenever agent/tools.py or prompts/tools.json changes.
---
# Add or change an agent tool
1. Name the failing question kind it fixes (label + question ids).
2. Schema in `prompts/tools.json`: name, a description (what it does, when to
   use it, an example, what comes back; a list of lines is joined with newlines),
   typed args with limits. Bump the file's `version`.
3. Logic in `retrieval/`; the adapter (a Pydantic arguments model + the result text) in
   `agent/corpus_tools.py`, registered in `CorpusToolSession`'s `ToolBox`; the box in
   `agent/tools.py` dispatches and turns every error into text. Deterministic: no LLM
   calls inside a tool. A mistake the model can fix raises `ToolInputError` with a
   message written for the model (it reaches the model; any other exception shows only
   its type).
4. Observation: compact, uses handles (D#, R#), never cut: a long list pages with
   "rows 1-50 of 278; next: offset=50" (user decision: nothing is dropped without the
   model being told and able to reach it).
5. Trace span: the box makes `tool:<name>` (arguments given and checked, the exact text);
   the tool adds its internals (`obs.current().update(...)` or its own span).
6. Tests: exact result on the fixture corpus (`tests/unit/tool_corpus.py`); a FakeLLM
   loop that calls it; bad arguments give an error observation; its span exists.
7. One tool-guide line in `prompts/agent.md` (when to use it; a test checks one line per
   tool). Bump both files' versions and add a `docs/prompt-log.md` entry.
8. `run-eval` smoke + tune; keep by the keep-rule.
