---
name: LLM (chat model) gotchas
description: Traps with Gemini chat calls through google-genai - a model turn rebuilt without thought signatures is refused with 400 (re-send raw, or put the stand-in signature on the first function call); the SDK sends two fields in snake_case on the wire; a genai.Client that nothing references closes its HTTP client ("client has been closed"). Embedding traps are in embeddings.md.
source: runs/2026-10-16-agent-tools
---

# LLM (chat model) gotchas

## 2026-10-16 A model turn rebuilt without thought signatures is refused
- **Area:** llm
- **Symptom:** `400 INVALID_ARGUMENT: Function call is missing a thought_signature in
  functionCall parts` when a model Content is rebuilt from the function calls alone.
- **Cause:** Gemini 3.x checks that the function calls of the current turn carry the
  signature the model gave them.
- **Fix:** re-send the model's own Content unchanged (`Message.raw`). A turn written by code
  (the first search) gets the documented stand-in `skip_thought_signature_validator` on its
  first function-call part (`CHAT_STAND_IN_THOUGHT_SIGNATURE`; the probe and tune runs T4-T5
  showed it is accepted).
- **Avoid next time:** never rebuild a model turn the model wrote; keep `raw`.

## 2026-10-16 google-genai sends two fields in snake_case on the wire
- **Area:** llm
- **Symptom:** the offline wire test failed: `KeyError: 'parametersJsonSchema'`, then
  `{'thinking_budget': 256} != {'thinkingBudget': 256}`.
- **Cause:** google-genai 2.31 sends `parameters_json_schema` and the thinking config's
  `thinking_budget` in snake_case (other fields in camelCase). The API accepts both.
- **Fix:** the test checks the keys the SDK really sends, with a comment saying why.
- **Avoid next time:** read the request body from a mock transport before asserting on field
  names.

## 2026-10-16 `genai.Client(...).models.list()` fails: "client has been closed"
- **Area:** llm
- **Symptom:** `RuntimeError: Cannot send a request, as the client has been closed.`
- **Cause:** nothing kept a reference to the `Client`; freeing it closes its HTTP client
  before the request is sent.
- **Fix:** keep the client in a variable (or on `self`, as `GeminiChat` does) for as long as
  calls use it.
- **Avoid next time:** never chain calls on a `genai.Client` that nothing holds on to.
