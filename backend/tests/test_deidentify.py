"""Inbound PHI de-identification: the round trip, and its limits.

Two things have to be true at once, and they pull against each other:

* the external model must never see a patient name, in the user's message OR in
  a tool result fed back on a later turn;
* tool calling must keep working, which means real names have to reappear in
  tool arguments before LibreChat executes them.

A scrubber that satisfies only the first breaks the product and gets switched
off, so both directions are tested.
"""

from __future__ import annotations

import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from guard import app as guard_app
from guard.deidentify import PhiRedactor, token_for

ROSTER = (
    "Maya Cohen",
    "David Levi",
    "Avraham Friedman",
    "Rivka Shapiro",
)


@pytest.fixture
def redactor() -> PhiRedactor:
    return PhiRedactor(names=ROSTER)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_token_is_stable_across_calls_and_processes() -> None:
    """Derived from the name, not allocated, so no shared session state.

    The model can still follow "the same patient" across turns without ever
    learning who it is.
    """
    assert token_for("Maya Cohen") == token_for("maya cohen")
    assert token_for("Maya Cohen") != token_for("David Levi")
    assert token_for("Maya Cohen").startswith("Patient Zx")


def test_token_does_not_contain_the_name() -> None:
    token = token_for("Maya Cohen")
    assert "maya" not in token.lower()
    assert "cohen" not in token.lower()


def test_token_contains_no_digits() -> None:
    """REGRESSION from a live Tier B run.

    The original `[PATIENT_7F3A2C]` read as a patient ID: the model skipped
    find_patient and "normalised" the token into the P### shape the tool
    description asks for, emitting get_current_biomarkers(patient_id="P084") —
    an identifier invented from the token's own hex digits. With no digits there
    is nothing to build a P### from.
    """
    for name in ROSTER:
        token = token_for(name)
        assert not any(ch.isdigit() for ch in token), token


def test_token_is_name_shaped_not_id_shaped() -> None:
    """It must route to find_patient the way any other name would."""
    token = token_for("Maya Cohen")
    assert token.startswith("Patient Zx")
    assert "[" not in token and "_" not in token
    # Must not look like the patient_id format the clinical tools expect.
    import re as _re

    assert not _re.fullmatch(r"P\d{3}", token)


def test_restore_is_case_and_whitespace_tolerant(redactor: PhiRedactor) -> None:
    """A model may re-type a token; an unrestored token is a failed tool call."""
    token = token_for("Maya Cohen")
    assert redactor.restore(token.lower()) == "Maya Cohen"
    assert redactor.restore(token.upper()) == "Maya Cohen"
    assert redactor.restore(token.replace(" ", "  ")) == "Maya Cohen"


def test_bare_token_core_restores(redactor: PhiRedactor) -> None:
    """REGRESSION from a live Tier B run.

    The model read "Patient Zxsyqn" as title + surname and called
    find_patient(name="Zxsyqn") — correct routing, but the bare core did not
    restore, so the lookup failed and it reported the patient as non-existent.
    """
    core = token_for("Maya Cohen").removeprefix("Patient ")
    assert redactor.restore(core) == "Maya Cohen"
    assert redactor.restore(core.lower()) == "Maya Cohen"

    call = {
        "id": "c1",
        "function": {"name": "find_patient", "arguments": json.dumps({"name": core})},
    }
    restored = redactor.restore_tool_call(call)
    assert json.loads(restored["function"]["arguments"])["name"] == "Maya Cohen"


def test_ordinary_prose_is_not_mistaken_for_a_token(redactor: PhiRedactor) -> None:
    """"Patient record" is six letters after "Patient" — it must survive."""
    for text in ("Patient record updated.", "The patient reports fatigue."):
        assert redactor.restore(text) == text


# ---------------------------------------------------------------------------
# Scrubbing
# ---------------------------------------------------------------------------


def test_full_name_is_scrubbed(redactor: PhiRedactor) -> None:
    out = redactor.scrub("What is Maya Cohen's most recent eGFR?")
    assert "Maya" not in out and "Cohen" not in out
    assert token_for("Maya Cohen") in out


@pytest.mark.parametrize(
    "text",
    [
        "What is Maya Cohen's eGFR?",
        "what is maya cohen's egfr?",
        "Tell me about COHEN, MAYA",
        "How is Cohen doing?",
        "Any update on Maya?",
    ],
)
def test_name_variants_are_all_caught(redactor: PhiRedactor, text: str) -> None:
    """Doctors write names however they like; a first-name-only mention leaks too."""
    out = redactor.scrub(text)
    assert "cohen" not in out.lower()
    assert "maya" not in out.lower()


def test_longest_variant_wins(redactor: PhiRedactor) -> None:
    """Replacing "Cohen" first would leave "Maya [PATIENT_...]".

    Half-scrubbed is worse than either extreme, because it looks safe.
    """
    out = redactor.scrub("Maya Cohen")
    assert out == token_for("Maya Cohen")
    assert out.count("Patient Zx") == 1


def test_unrelated_words_are_untouched(redactor: PhiRedactor) -> None:
    """Over-scrubbing would mangle clinical text and get the guard disabled."""
    text = "The Framingham CKD score uses eGFR and proteinuria. CAIDE covers dementia."
    assert redactor.scrub(text) == text


def test_substrings_are_not_scrubbed(redactor: PhiRedactor) -> None:
    """Whole-word matching: "Levine" must not be mangled because "Levi" exists."""
    assert "Levine" in redactor.scrub("Dr Levine reviewed the chart.")


# ---------------------------------------------------------------------------
# Restoring
# ---------------------------------------------------------------------------


def test_round_trip_is_lossless(redactor: PhiRedactor) -> None:
    original = "Compare Maya Cohen and David Levi."
    assert redactor.restore(redactor.scrub(original)) == original


def test_restore_leaves_unknown_tokens_alone(redactor: PhiRedactor) -> None:
    assert redactor.restore("[PATIENT_ABCDEF] is unknown") == "[PATIENT_ABCDEF] is unknown"


def test_tool_arguments_are_restored(redactor: PhiRedactor) -> None:
    """Without this the model emits find_patient("[PATIENT_...]") and lookup fails.

    MCP is inside the trust boundary; it should see the real name.
    """
    call = {
        "id": "call_1",
        "function": {
            "name": "find_patient",
            "arguments": json.dumps({"name": token_for("Maya Cohen")}),
        },
    }
    restored = redactor.restore_tool_call(call)
    assert json.loads(restored["function"]["arguments"])["name"] == "Maya Cohen"


# ---------------------------------------------------------------------------
# Whole-conversation scrubbing
# ---------------------------------------------------------------------------


def test_tool_results_are_scrubbed_too(redactor: PhiRedactor) -> None:
    """The leak that a user-message-only scrubber misses.

    MCP returns the real name in the tool result, and LibreChat feeds it straight
    back into the next request.
    """
    messages = [
        {"role": "user", "content": "What is Maya Cohen's eGFR?"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({"name": "Maya Cohen", "egfr_ml_min_1_73m2": 102.0}),
        },
    ]
    cleaned, _ = redactor.scrub_messages(messages)
    blob = json.dumps(cleaned)
    assert "Maya" not in blob and "Cohen" not in blob
    assert "102.0" in blob, "clinical values are not PHI and must survive"


def test_assistant_tool_calls_are_scrubbed_on_the_way_out(redactor: PhiRedactor) -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "find_patient",
                        "arguments": json.dumps({"name": "Maya Cohen"}),
                    },
                }
            ],
        }
    ]
    cleaned, _ = redactor.scrub_messages(messages)
    assert "Cohen" not in json.dumps(cleaned)


def test_inactive_redactor_is_a_no_op() -> None:
    empty = PhiRedactor()
    assert not empty.active
    assert empty.scrub("Maya Cohen") == "Maya Cohen"


# ---------------------------------------------------------------------------
# End to end through the proxy
# ---------------------------------------------------------------------------


@pytest.fixture
def phi_guard_client():
    async def _make(handler) -> AsyncClient:
        upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://upstream.test/v1"
        )
        guard_app.app.state.upstream = upstream
        guard_app.app.state.redactor = PhiRedactor(names=ROSTER)
        # Far in the future so the roster is not re-fetched over the network.
        guard_app.app.state.phi_loaded_at = 1e18
        return AsyncClient(
            transport=ASGITransport(app=guard_app.app), base_url="http://guard"
        )

    return _make


async def test_upstream_never_receives_a_patient_name(phi_guard_client) -> None:
    """The whole point: OpenRouter sees a token, never "Maya Cohen"."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"{token_for('Maya Cohen')}'s eGFR is 102.0.",
                        },
                    }
                ]
            },
        )

    async with await phi_guard_client(handler) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "What is Maya Cohen's eGFR?"}],
            },
        )

    assert "Maya" not in seen["body"] and "Cohen" not in seen["body"]
    assert token_for("Maya Cohen") in seen["body"]

    # ...and the clinician still reads a real name.
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("Maya Cohen's eGFR is 102.0")


async def test_tool_calls_come_back_with_real_names(phi_guard_client) -> None:
    """Tool calling must survive the boundary or the feature is unusable."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "find_patient",
                                        "arguments": json.dumps(
                                            {"name": token_for("Avraham Friedman")}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    async with await phi_guard_client(handler) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Risks for Avraham Friedman?"}],
            },
        )

    arguments = response.json()["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert json.loads(arguments)["name"] == "Avraham Friedman"


async def test_streaming_also_restores_names(phi_guard_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        token = token_for("Rivka Shapiro")
        chunks = [
            "data: "
            + json.dumps(
                {
                    "id": "1",
                    "model": "m",
                    "choices": [{"index": 0, "delta": {"content": piece}}],
                }
            )
            for piece in (f"{token}'s ", "dementia risk is 0.45.")
        ]
        return httpx.Response(
            200,
            text="\n\n".join(chunks) + "\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    async with await phi_guard_client(handler) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "Rivka Shapiro's dementia risk?"}],
            },
        )
        text = response.text

    delivered = "".join(
        json.loads(line[5:]).get("choices", [{}])[0].get("delta", {}).get("content", "")
        for line in text.splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    )
    assert delivered.startswith("Rivka Shapiro's dementia risk is 0.45")
    assert "Patient Zx" not in delivered
