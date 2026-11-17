"""Live probe of the Gemini chat behaviour the agent harness relies on (plan step 1).

It answers, with real calls, what `GeminiChat` (plan section 4) must handle: several
function calls in one reply, mode ANY limited to `finish`, re-sending the model's own reply
(thought signatures), two user turns in a row, a working-memory text before a kept tool
step, usage field names, latency and cost. Every call is logged on one line (run with
`uv run tests-live -k chat_probe -o log_cli=true --log-cli-level=INFO`).

Marked live: skipped by default. Prompts are tiny and invented (nothing from data/). The
probe's own spend ledger (in a tmp folder) refuses any call that would pass
PROBE_CAP_USD. Required checks fail the test; info checks only log what happened.
"""

import asyncio
import base64
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from google import genai
from google.genai import types

from corpusqa.llm.limits import SpendLedger
from corpusqa.prompts import load_tool_schemas

pytestmark = pytest.mark.live
logger = logging.getLogger(__name__)

FLASH = "gemini-3.8-flash"
PRO_CLASS = ("gemini-3.1-pro-preview", "gemini-4-argon")
# Logged with their limits by the model-list test.
CANDIDATES = frozenset(
    {
        FLASH,
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-pro-latest",
        *PRO_CLASS,
    }
)
# USD per 1M tokens (input, cached input, output incl. thinking): paid tier, Standard,
# prompts <= 200k tokens, from ai.google.dev/gemini-api/docs/pricing (read 2026-10-16;
# the page says "Last updated 2026-10-15 UTC"). gemini-3.8-flash doubles on 2027-01-01,
# gemini-4-argon on 2026-12-01.
PRICES = {
    "gemini-3.8-flash": (0.75, 0.075, 3.75),
    "gemini-3.1-pro-preview": (2.00, 0.20, 12.00),
    "gemini-4-argon": (2.00, 0.10, 10.00),
}
# The cap check reserves each call at the dearest listed price (argon from 2026-12-01).
MAX_PRICES = (4.00, 0.20, 20.00)
PROBE_CAP_USD = 0.50
TOKEN_CHARS = 4
MAX_OUTPUT_TOKENS = 2_048
THINKING_BUDGET = 512
HTTP_TIMEOUT_MS = 90_000

SYSTEM = "You are a test assistant. Use the tools exactly as the user asks. Keep text short."
QUESTION = (
    "Look up the words 'quokka' and 'axolotl' with the lookup tool. "
    "Make both calls in this one reply."
)
WORKING_MEMORY = (
    "<working_memory>\nEarlier steps looked up 'kiwi': a flightless bird. Still open: "
    "'quokka' and 'axolotl'.\n</working_memory>"
)
DICTIONARY = {"quokka": "a small wallaby", "axolotl": "a salamander that keeps its gills"}
# The documented stand-in signature, sent two ways: as these bytes, and as the bytes whose
# base64 is this text (the API field is base64).
DUMMY_SIGNATURE_TEXT = "skip_thought_signature_validator"
DUMMY_SIGNATURES = {
    "raw_bytes": DUMMY_SIGNATURE_TEXT.encode("ascii"),
    "base64_text": base64.urlsafe_b64decode(
        DUMMY_SIGNATURE_TEXT + "=" * (-len(DUMMY_SIGNATURE_TEXT) % 4)
    ),
}
HARD_PROMPT = (
    "A train leaves at 09:47 and arrives at 14:13 the same day, with two stops of 7 and "
    "11 minutes. How many minutes was it moving? Answer with the number only."
)
HARD_PROMPT_THINKING = {
    "budget_1024": types.ThinkingConfig(thinking_budget=1_024),
    "budget_dynamic": types.ThinkingConfig(thinking_budget=-1),
    "level_high": types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
}
LOOKUP = types.FunctionDeclaration(
    name="lookup",
    description="Look up the meaning of one word in a small dictionary. Returns the meaning.",
    parameters_json_schema={
        "type": "object",
        "properties": {"word": {"type": "string", "description": "One word, lower case."}},
        "required": ["word"],
        "additionalProperties": False,
    },
)


def finish_declaration() -> types.FunctionDeclaration:
    """The shipped `finish` schema from prompts/tools.json, as Gemini will get it."""
    schema = load_tool_schemas().get("finish")
    return types.FunctionDeclaration(
        name=schema.name,
        description=schema.description,
        parameters_json_schema=dict(schema.parameters),
    )


def user(*texts: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text) for text in texts])


def calls_of(response: types.GenerateContentResponse) -> list[types.FunctionCall]:
    return list(response.function_calls or [])


def raw_of(response: types.GenerateContentResponse) -> types.Content:
    assert response.candidates and response.candidates[0].content, "no candidate content"
    return response.candidates[0].content


def answers(calls: list[types.FunctionCall]) -> types.Content:
    """One user Content with one function_response per call, ids kept."""
    parts = [
        types.Part(
            function_response=types.FunctionResponse(
                id=call.id,
                name=call.name,
                response={"result": DICTIONARY.get(str((call.args or {}).get("word")), "unknown")},
            )
        )
        for call in calls
    ]
    return types.Content(role="user", parts=parts)


def without_signatures(raw: types.Content, first_signature: bytes | None = None) -> types.Content:
    """The model turn rebuilt from its calls alone, as code would build it with no `raw`;
    optionally with a stand-in signature on the first call (ai.google.dev thought-signatures
    docs name a dummy value for turns the model did not write)."""
    calls = [part.function_call for part in raw.parts or [] if part.function_call]
    parts = [
        types.Part(
            function_call=types.FunctionCall(id=call.id, name=call.name, args=call.args),
            thought_signature=first_signature if i == 0 else None,
        )
        for i, call in enumerate(calls)
    ]
    return types.Content(role="model", parts=parts)


def describe(response: types.GenerateContentResponse) -> str:
    candidate = (response.candidates or [None])[0]
    if candidate is None:
        return f"no candidate (prompt_feedback={response.prompt_feedback})"
    parts = []
    for part in (candidate.content.parts if candidate.content else None) or []:
        sig = "+sig" if part.thought_signature else ""
        if part.thought:
            parts.append(f"thought({len(part.text or '')}c){sig}")
        elif part.function_call:
            fc = part.function_call
            parts.append(f"call {fc.name}({dict(fc.args or {})}) id={fc.id!r}{sig}")
        elif part.text is not None:
            parts.append(f"text({part.text[:60]!r}){sig}")
        else:
            parts.append(f"other{sig}")
    return f"finish={candidate.finish_reason} parts=[{'; '.join(parts)}]"


@dataclass
class Probe:
    client: genai.Client
    ledger: SpendLedger
    results: list[tuple[str, bool, bool, str]] = field(default_factory=list)

    async def ask(
        self,
        label: str,
        model: str,
        contents: list[types.Content],
        *,
        tools: list[types.FunctionDeclaration] | None = None,
        mode: str = "AUTO",
        allowed: list[str] | None = None,
        thinking: types.ThinkingConfig | None = None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> types.GenerateContentResponse:
        tools = [LOOKUP, finish_declaration()] if tools is None else tools
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=tools)] if tools else None,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode(mode), allowed_function_names=allowed
                )
            )
            if tools
            else None,
            thinking_config=thinking or types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            max_output_tokens=max_output_tokens,
        )
        sent_chars = len(str(contents)) + len(SYSTEM) + len(str(tools))
        reserved = cost(MAX_PRICES, sent_chars // TOKEN_CHARS, 0, max_output_tokens)
        self.ledger.reserve(reserved)
        started = time.monotonic()
        try:
            response = await self.client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            self.ledger.release(reserved)  # an error answer is not billed
            logger.info(
                "CALL %s | %s | %.2f s | ERROR %s",
                label,
                model,
                time.monotonic() - started,
                str(exc)[:300],
            )
            raise
        latency = time.monotonic() - started
        usage = response.usage_metadata or types.GenerateContentResponseUsageMetadata()
        out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        spent = cost(
            PRICES[model],
            usage.prompt_token_count or 0,
            usage.cached_content_token_count or 0,
            out,
        )
        self.ledger.add((usage.prompt_token_count or 0) + out, spent, reserved_usd=reserved)
        logger.info(
            "CALL %s | %s | %.2f s | %s | usage=%s | $%.5f",
            label,
            model,
            latency,
            describe(response),
            usage.model_dump(exclude_none=True, mode="json"),
            spent,
        )
        return response

    async def check(
        self, name: str, run: Callable[[], Awaitable[str]], required: bool = True
    ) -> None:
        try:
            detail = await run()
            ok = True
        except Exception as exc:  # a failed check is a finding: log it, go on
            detail = f"{type(exc).__name__}: {str(exc)[:400]}"
            ok = False
        self.results.append((name, ok, required, detail))
        kind = "required" if required else "info"
        logger.info("CHECK %s (%s): %s %s", name, kind, "OK" if ok else "FAIL", detail)

    def assert_required(self) -> None:
        failed = [
            f"{name}: {detail}"
            for name, ok, required, detail in self.results
            if required and not ok
        ]
        assert not failed, "required checks failed:\n" + "\n".join(failed)


def cost(prices: tuple[float, float, float], prompt: int, cached: int, out: int) -> float:
    price_in, price_cached, price_out = prices
    return ((prompt - cached) * price_in + cached * price_cached + out * price_out) / 1e6


def live_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    return genai.Client(
        vertexai=False, api_key=key, http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS)
    )


@pytest.fixture(scope="module")
def ledger(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SpendLedger]:
    path: Path = tmp_path_factory.mktemp("probe") / "spend.json"
    spend = SpendLedger(path, PROBE_CAP_USD)
    yield spend
    totals = spend.totals
    logger.info("SPEND %d calls, %d tokens, $%.4f", totals.calls, totals.tokens, totals.cost_usd)


# ---------------------------------------------------------------- the checks
def test_live_chat_probe_lists_generation_models() -> None:
    client = live_client()  # keep a reference: a dropped Client closes its HTTP client
    models = list(client.models.list())
    rows = sorted(
        (m.name or "", ",".join(m.supported_actions or []), str(m.thinking))
        for m in models
        if "generateContent" in (m.supported_actions or [])
    )
    for name, actions, thinking in rows:
        logger.info("MODEL %s | thinking=%s | %s", name, thinking, actions)
    for m in models:
        if (m.name or "").removeprefix("models/") in CANDIDATES:
            logger.info(
                "DETAIL %s | %s | in=%s out=%s | %s | %s",
                m.name,
                m.display_name,
                m.input_token_limit,
                m.output_token_limit,
                m.version,
                (m.description or "")[:200],
            )
    assert rows


def test_live_chat_probe_flash_behaviour(ledger: SpendLedger) -> None:
    probe = Probe(live_client(), ledger)
    asyncio.run(flash_checks(probe, FLASH))
    probe.assert_required()


async def flash_checks(probe: Probe, model: str) -> None:
    kept: dict[str, types.GenerateContentResponse] = {}

    async def parallel_calls() -> str:
        reply = await probe.ask("parallel", model, [user(QUESTION)])
        kept["first"] = reply
        calls = calls_of(reply)
        assert len(calls) >= 2, f"{len(calls)} function calls: {describe(reply)}"
        return f"{len(calls)} calls; ids={[c.id for c in calls]}"

    def first_step() -> tuple[types.Content, types.Content]:
        reply = kept["first"]
        return raw_of(reply), answers(calls_of(reply))

    async def resend_raw() -> str:
        raw, results = first_step()
        reply = await probe.ask("resend_raw", model, [user(QUESTION), raw, results])
        return describe(reply)

    async def any_finish_only() -> str:
        reply = await probe.ask(
            "any_finish",
            model,
            [user("What is 2 + 2? Reply by calling a tool. There are no documents to cite.")],
            mode="ANY",
            allowed=["finish"],
        )
        calls = calls_of(reply)
        assert [c.name for c in calls] == ["finish"], describe(reply)
        args = dict(calls[0].args or {})
        assert set(args) == {"answer", "citations", "not_found"}, args
        return f"args={args}"

    async def two_user_turns() -> str:
        reply = await probe.ask("two_user", model, [user(QUESTION), user(WORKING_MEMORY)])
        return describe(reply)

    async def memory_before_kept_step_merged() -> str:
        raw, results = first_step()
        reply = await probe.ask(
            "memory_merged", model, [user(QUESTION, WORKING_MEMORY), raw, results]
        )
        return describe(reply)

    async def memory_before_kept_step_separate() -> str:
        raw, results = first_step()
        reply = await probe.ask(
            "memory_separate", model, [user(QUESTION), user(WORKING_MEMORY), raw, results]
        )
        return describe(reply)

    async def rebuilt_without_signatures() -> str:
        raw, results = first_step()
        rebuilt = without_signatures(raw)
        reply = await probe.ask("no_signature", model, [user(QUESTION), rebuilt, results])
        return describe(reply)

    async def rebuilt_with_dummy_signature(signature: bytes) -> str:
        raw, results = first_step()
        rebuilt = without_signatures(raw, first_signature=signature)
        reply = await probe.ask("dummy_signature", model, [user(QUESTION), rebuilt, results])
        return describe(reply)

    async def thinks_on_a_hard_prompt(thinking: types.ThinkingConfig) -> str:
        reply = await probe.ask(
            "hard_prompt",
            model,
            [user(HARD_PROMPT)],
            tools=[],
            thinking=thinking,
        )
        usage = reply.usage_metadata or types.GenerateContentResponseUsageMetadata()
        return f"{thinking} -> thoughts={usage.thoughts_token_count}; {describe(reply)}"

    async def thinking_level_low() -> str:
        reply = await probe.ask(
            "thinking_level",
            model,
            [user("What is the capital of France? One word.")],
            tools=[],
            thinking=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        )
        return describe(reply)

    async def max_tokens_counts_thinking() -> str:
        reply = await probe.ask(
            "max_tokens",
            model,
            [user("Is 391 prime? Work it out carefully, then answer yes or no with factors.")],
            tools=[],
            thinking=types.ThinkingConfig(thinking_budget=1_024),
            max_output_tokens=128,
        )
        usage = reply.usage_metadata or types.GenerateContentResponseUsageMetadata()
        return (
            f"max_output_tokens=128 -> thoughts={usage.thoughts_token_count} "
            f"output={usage.candidates_token_count}; {describe(reply)}"
        )

    await probe.check("parallel_calls", parallel_calls)
    await probe.check("resend_raw_unchanged", resend_raw)
    await probe.check("any_mode_only_finish", any_finish_only)
    await probe.check("two_user_contents_in_a_row", two_user_turns)
    await probe.check("memory_before_kept_step_merged", memory_before_kept_step_merged)
    await probe.check(
        "memory_before_kept_step_separate", memory_before_kept_step_separate, required=False
    )
    await probe.check(
        "model_turn_rebuilt_without_signatures", rebuilt_without_signatures, required=False
    )
    for name, signature in DUMMY_SIGNATURES.items():
        await probe.check(
            f"model_turn_rebuilt_with_dummy_signature_{name}",
            lambda s=signature: rebuilt_with_dummy_signature(s),
            required=False,
        )
    for name, thinking in HARD_PROMPT_THINKING.items():
        await probe.check(
            f"thinks_on_a_hard_prompt_{name}",
            lambda t=thinking: thinks_on_a_hard_prompt(t),
            required=False,
        )
    await probe.check("thinking_level_low", thinking_level_low, required=False)
    await probe.check(
        "max_output_tokens_counts_thinking", max_tokens_counts_thinking, required=False
    )


@pytest.mark.parametrize("model", PRO_CLASS)
def test_live_chat_probe_pro_class(ledger: SpendLedger, model: str) -> None:
    probe = Probe(live_client(), ledger)
    asyncio.run(pro_checks(probe, model))
    probe.assert_required()


async def pro_checks(probe: Probe, model: str) -> None:
    kept: dict[str, types.GenerateContentResponse] = {}

    async def parallel_calls() -> str:
        reply = await probe.ask("parallel", model, [user(QUESTION)])
        kept["first"] = reply
        calls = calls_of(reply)
        assert len(calls) >= 2, f"{len(calls)} function calls: {describe(reply)}"
        return f"{len(calls)} calls"

    async def resend_raw() -> str:
        reply = kept["first"]
        contents = [user(QUESTION), raw_of(reply), answers(calls_of(reply))]
        return describe(await probe.ask("resend_raw", model, contents))

    await probe.check("parallel_calls", parallel_calls)
    await probe.check("resend_raw_unchanged", resend_raw)
