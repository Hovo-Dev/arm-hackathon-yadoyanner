"""Thin wrapper over the DeepSeek reasoning core (via the OpenRouter gateway).

Deliberately small: the agent owns the research logic, this owns transport,
retries, JSON coercion and cost accounting.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI, APIError

from . import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class UsageMeter:
    """Tracks tokens across a whole research run so cost is reportable."""

    def __init__(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0

    def add(self, usage: Any) -> None:
        if not usage:
            return
        self.tokens_in += getattr(usage, "prompt_tokens", 0) or 0
        self.tokens_out += getattr(usage, "completion_tokens", 0) or 0
        self.calls += 1

    @property
    def est_cost_usd(self) -> float:
        return (
            self.tokens_in / 1_000_000 * config.PRICE_IN_PER_M
            + self.tokens_out / 1_000_000 * config.PRICE_OUT_PER_M
        )

    def __str__(self) -> str:
        return (
            f"{self.calls} calls | in={self.tokens_in} out={self.tokens_out} "
            f"| ~${self.est_cost_usd:.4f}"
        )


class ReasoningClient:
    """DeepSeek through OpenRouter. Text-only — images must be captioned first."""

    def __init__(self, model: str | None = None, meter: UsageMeter | None = None):
        if not config.OPENROUTER_API_KEY:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. Put it in .env at the repo root."
            )
        self.model = model or config.REASONING_MODEL
        self.meter = meter or UsageMeter()
        self._client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
        )

    # -- core call ---------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        retries: int = 3,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                self.meter.add(getattr(resp, "usage", None))
                return resp.choices[0].message
            except APIError as exc:  # rate limits, 5xx, transient gateway errors
                last_err = exc
                wait = 2**attempt
                log.warning("LLM call failed (attempt %d): %s — retrying in %ss",
                            attempt + 1, exc, wait)
                time.sleep(wait)
            except Exception as exc:  # noqa: BLE001 - surface anything else
                raise LLMError(f"LLM call failed: {exc}") from exc
        raise LLMError(f"LLM call failed after {retries} attempts: {last_err}")

    # -- JSON helper -------------------------------------------------------
    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 3000,
    ) -> dict[str, Any]:
        """Ask for JSON and actually get JSON back.

        DeepSeek honours response_format most of the time but still occasionally
        wraps output in prose or a ``` fence, so we salvage rather than crash —
        a research run should not die on a stray backtick.
        """
        msg = self.chat(
            messages + [{
                "role": "system",
                "content": "Reply with a single valid JSON object and nothing else.",
            }],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _coerce_json(msg.content or "")


def _coerce_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            text = fenced.group(1).strip()

    # Widest brace span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Truncated output — recover the complete prefix rather than losing the run.
    if start != -1:
        repaired = _repair_truncated_json(text[start:])
        if repaired is not None:
            log.warning("Model output was truncated; salvaged the complete prefix.")
            return repaired

    log.warning("Could not parse JSON from model output (%d chars)", len(text))
    return {}


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Close a JSON document that ran out of output tokens mid-write.

    A long report can exceed max_tokens partway through an array element. The
    complete elements before the cut are still perfectly good, so we trim back
    to the last finished one and close the open brackets rather than throwing
    away an entire (paid-for) research run.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    safe_cut = -1

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if stack:  # a nested element just closed cleanly
                safe_cut = i

    if not stack:
        return None
    if safe_cut < 0:
        return None

    candidate = text[: safe_cut + 1]

    # Recompute what is still open on the trimmed candidate, then close it.
    depth: list[str] = []
    in_string = False
    escaped = False
    for ch in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth.append(ch)
        elif ch in "}]" and depth:
            depth.pop()

    closers = "".join("}" if c == "{" else "]" for c in reversed(depth))
    try:
        return json.loads(candidate + closers)
    except json.JSONDecodeError:
        return None
