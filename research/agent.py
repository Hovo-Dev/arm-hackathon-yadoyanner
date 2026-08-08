"""The deep-research agent.

Shape of a run:

    intake  -> normalise free text into a structured problem
    research-> bounded tool-calling loop, accumulating cited evidence
    synth   -> ranked causes + parts + next steps, every claim citing evidence

DeepSeek has no built-in web search, so this loop *is* the deep research. That
is the component worth building well, and it is provider-independent: when a
Perplexity key appears it becomes one more tool, not a rewrite.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from typing import Any

from . import config, prompts
from .llm import LLMError, ReasoningClient, UsageMeter
from .schemas import (
    CarProblem,
    Cause,
    DiagnosticStep,
    Evidence,
    PartSuggestion,
    ResearchEvent,
    ResearchReport,
    VehicleContext,
)
from .sources import obd, vpic
from .tools import (
    PROVIDER_FAILURES,
    available_tool_schemas,
    build_dispatcher,
    execute_tool_call,
    summarise_for_model,
)

log = logging.getLogger(__name__)


def _titlecase(value: str) -> str:
    """'nissan' -> 'Nissan', 'x-trail' -> 'X-Trail'. Leaves 'BMW' alone."""
    if not value or value.isupper():
        return value
    return "-".join(part.capitalize() for part in value.split("-"))


_NON_LATIN = re.compile(r"[԰-֏Ѐ-ӿ]")  # Armenian, Cyrillic
_SYMPTOM_SPLIT = re.compile(r"\s*(?:[,;.]|\band\b|\balso\b|\n)\s*", re.I)


def _deterministic_intake(raw_text: str, vehicle: VehicleContext) -> CarProblem:
    """Build the problem without an LLM call.

    Valid when the UI already collected make/model/year and the text is Latin
    script. In that case the model was being asked to restate an English
    sentence as a list — 12.8 seconds and an API call to turn "brakes squeal,
    worse in the wet" into ["brakes squeal", "worse in the wet"].

    Non-Latin input still needs the model: symptoms must reach the NHTSA search
    in English, and translating "ջրերը եռցնումա" is real work, not restatement.
    """
    parts = [p.strip() for p in _SYMPTOM_SPLIT.split(raw_text) if len(p.strip()) > 3]
    return CarProblem(
        raw_text=raw_text,
        vehicle=vehicle,
        symptoms=parts[:6],
        obd_codes=obd.extract_codes(raw_text),
        language="en",
    )


def can_skip_intake_llm(raw_text: str, hint: VehicleContext | None) -> bool:
    """True when the vehicle is known and the text needs no translation."""
    return bool(
        hint and hint.make and hint.model
        and not _NON_LATIN.search(raw_text or "")
    )


def intake(raw_text: str, client: ReasoningClient,
           hint: VehicleContext | None = None) -> CarProblem:
    """Free text (+ optional known vehicle) -> structured CarProblem.

    When the caller already supplied make/model/year — the normal path for a UI
    with vehicle dropdowns — the model is asked only for symptoms and codes, not
    to re-derive facts we were handed. The query text in that flow ("brakes
    squeal in the wet") contains no vehicle at all, so asking it to extract one
    invites a guess.
    """
    if can_skip_intake_llm(raw_text, hint):
        log.info("Intake: deterministic (vehicle known, Latin script) — no LLM call")
        return _deterministic_intake(raw_text, hint)

    have_vehicle = bool(hint and hint.make and hint.model)

    data = client.chat_json(
        [
            {"role": "system", "content": prompts.INTAKE},
            {
                "role": "user",
                "content": (
                    f"Vehicle is already known: {hint.label()}. Extract only "
                    f"symptoms, codes and language.\n\n{raw_text}"
                    if have_vehicle
                    else raw_text
                ),
            },
        ]
    )

    vehicle = VehicleContext(
        make=_titlecase((hint.make if hint else None) or data.get("make") or "unknown"),
        model=_titlecase((hint.model if hint else None) or data.get("model") or "unknown"),
        year=(hint.year if hint else None) or data.get("year"),
        engine=(hint.engine if hint else None) or data.get("engine"),
        mileage_km=(hint.mileage_km if hint else None) or data.get("mileage_km"),
    )

    # Regex catches codes the model may have missed; union of both.
    codes = sorted(set(data.get("obd_codes") or []) | set(obd.extract_codes(raw_text)))

    return CarProblem(
        raw_text=raw_text,
        vehicle=vehicle,
        symptoms=data.get("symptoms") or [],
        obd_codes=codes,
        language=data.get("language") or "en",
    )


def _research_loop(
    problem: CarProblem,
    client: ReasoningClient,
    max_rounds: int,
) -> tuple[list[Evidence], list[str], int]:
    """Bounded tool-calling loop. Returns (evidence, tool_call_log, rounds)."""
    dispatcher = build_dispatcher(problem.vehicle)
    photo_block = (
        f"Findings from user photos: {'; '.join(problem.image_findings)}\n"
        if problem.image_findings
        else ""
    )

    schemas = available_tool_schemas(problem.vehicle, problem.obd_codes)
    source_block = prompts.source_block(
        [f"{s['function']['name']}: {s['function']['description']}" for s in schemas]
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": prompts.RESEARCH.format(
                vehicle=problem.vehicle.label(),
                problem=problem.raw_text,
                symptoms=", ".join(problem.symptoms) or "none stated",
                codes=", ".join(problem.obd_codes) or "none",
                photo_block=photo_block,
                source_block=source_block,
            ),
        },
        {"role": "user", "content": "Begin gathering evidence."},
    ]

    evidence: list[Evidence] = []
    call_log: list[str] = []
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        msg = client.chat(messages, tools=schemas, max_tokens=1200)
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            content = (msg.content or "").strip()
            log.info("Round %d: model stopped calling tools (%s)",
                     rounds, content[:80] or "empty reply")
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            name = tc.function.name
            call_log.append(name)
            log.info("Round %d: -> %s(%s)", rounds, name,
                     (tc.function.arguments or "")[:120])

            found = execute_tool_call(name, tc.function.arguments, dispatcher)
            room = config.MAX_EVIDENCE_ITEMS - len(evidence)
            found = found[: max(room, 0)]
            offset = len(evidence)
            evidence.extend(found)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": summarise_for_model(found, offset),
            })

        if len(evidence) >= config.MAX_EVIDENCE_ITEMS:
            log.info("Evidence cap (%d) reached — stopping collection.",
                     config.MAX_EVIDENCE_ITEMS)
            break

    return evidence, call_log, rounds


# Ceilings for what reaches the synthesis prompt. Evidence items differ in size
# by more than an order of magnitude — an NHTSA complaint is ~700 chars, a
# manual section up to 6,000 — so capping the COUNT (MAX_EVIDENCE_ITEMS) does
# not bound the prompt. An overheating run gathered 26 items totalling ~30,000
# characters and the report came back empty: the model exhausted its output
# budget mid-JSON. It succeeded on the retry, which is worse than a hard
# failure, because a flaky report fails during a demo rather than in testing.
SYNTH_CHAR_BUDGET = 18_000
SYNTH_ITEM_CHARS = 2_000


def _render_for_synthesis(evidence: list[Evidence]) -> str:
    """Render evidence within budget, keeping ORIGINAL indices.

    The index shown to the model must stay equal to its position in
    `report.evidence`, because that is the whole citation contract. Filtering
    into a new list and renumbering would silently repoint every citation — a
    cause citing [3] would resolve to a different source than the model read.
    So items are skipped, never renumbered.
    """
    lines: list[str] = []
    used = 0
    skipped = 0

    for i, ev in enumerate(evidence):
        # Navigation aids are not findings. The 197-heading manual contents
        # listing consumed 4,400 characters while telling the diagnosis nothing.
        if ev.raw.get("navigation_only"):
            skipped += 1
            continue

        snippet = ev.snippet
        if len(snippet) > SYNTH_ITEM_CHARS:
            snippet = snippet[: SYNTH_ITEM_CHARS - 1] + "…"
        if used + len(snippet) > SYNTH_CHAR_BUDGET:
            skipped += 1
            continue

        used += len(snippet)
        loc = f" <{ev.url}>" if ev.url else ""
        lines.append(f"[{i}] ({ev.source_key}) {ev.title}{loc}\n    {snippet}")

    if skipped:
        log.info("Synthesis: %d/%d items rendered (%d chars, %d skipped)",
                 len(lines), len(evidence), used, skipped)
    return "\n".join(lines)


def _synthesise(
    problem: CarProblem,
    evidence: list[Evidence],
    client: ReasoningClient,
) -> dict[str, Any]:
    rendered = _render_for_synthesis(evidence) if evidence else "(none gathered)"
    return client.chat_json(
        [
            {
                "role": "system",
                "content": prompts.SYNTHESIS.format(
                    vehicle=problem.vehicle.label(),
                    problem=problem.raw_text,
                    evidence=rendered,
                ),
            },
            {"role": "user", "content": "Produce the report."},
        ],
        max_tokens=5500,
    )


def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
    """Drop repeats, keeping the highest-scored copy.

    The model re-queries the same tool across rounds as it narrows in, so the
    same complaint arrives several times. Duplicates waste synthesis context
    and skew any 'how much evidence supports this' reading of the report.
    """
    best: dict[tuple[str, str], Evidence] = {}
    for ev in evidence:
        key = (ev.source_key, ev.title)
        current = best.get(key)
        if current is None or ev.relevance > current.relevance:
            best[key] = ev
    return sorted(best.values(), key=lambda e: e.relevance, reverse=True)


_PRICE_RE = re.compile(r"(\d[\d,\s]{2,9})\s*(?:AMD|֏|dram|драм)", re.I)
_YEAR_RANGE_RE = re.compile(r"\b(19|20)(\d{2})\s*[-–—]\s*(19|20)?(\d{2})\b")


def _digits(text: str) -> str:
    return re.sub(r"[,\s֏]", "", text)


def _fitment_years(fits: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for entry in fits:
        for m in _YEAR_RANGE_RE.finditer(str(entry)):
            start = int(m.group(1) + m.group(2))
            end_prefix = m.group(3) or m.group(1)
            end = int(end_prefix + m.group(4))
            if end >= start:
                ranges.append((start, end))
    return ranges


def _verify_part_claims(
    parts: list[PartSuggestion],
    evidence: list[Evidence],
    vehicle: VehicleContext,
) -> list[str]:
    """Check that what a part claims matches the listing it cites.

    Citation resolution only proves an index exists. It does not stop the model
    citing a real listing and then describing it wrongly — in testing, a listing
    stating "fits L33 2013-2015, L32 2007-2012" at 29,000 AMD was reported as
    "J31 2003-2008 / L31 2002-2005, fits 2004 Altima" at 32,000 AMD. Both the
    fitment and the price were invented on top of a real citation, and the
    grounding check passed it.

    Two independent claims are worth machine-checking because both cost money
    when wrong: the price, and whether the part fits this car at all.
    """
    issues: list[str] = []

    for part in parts:
        cited = [evidence[i] for i in part.evidence_ids if 0 <= i < len(evidence)]
        if not cited:
            continue
        blob = _digits(" ".join(f"{e.title} {e.snippet}" for e in cited))
        claim_text = f"{part.name} {part.notes or ''}"

        # 1. Any price quoted must appear verbatim in a cited source.
        for raw_price in _PRICE_RE.findall(claim_text):
            if _digits(raw_price) not in blob:
                msg = (
                    f"price {raw_price.strip()} AMD is not present in the cited "
                    f"listing(s) for '{part.name}'"
                )
                part.claim_issues.append(msg)
                issues.append(msg)

        # 2. Stated fitment must actually cover this vehicle's year.
        declared: list[str] = []
        for ev in cited:
            fits = (ev.raw.get("fields") or {}).get("compatible_models")
            if fits:
                declared.extend(str(f) for f in fits)

        if not declared or not vehicle.year:
            part.fitment_status = "unstated"
            continue

        ranges = _fitment_years(declared)
        if not ranges:
            part.fitment_status = "unstated"
            continue

        if any(lo <= vehicle.year <= hi for lo, hi in ranges):
            part.fitment_status = "verified"
        else:
            part.fitment_status = "mismatch"
            span = ", ".join(f"{lo}-{hi}" for lo, hi in ranges)
            msg = (
                f"FITMENT MISMATCH: '{part.name}' cites a listing covering {span}, "
                f"which does not include this vehicle's year ({vehicle.year})"
            )
            part.claim_issues.append(msg)
            issues.append(msg)

    return issues


def _coerce_report(
    data: dict[str, Any],
    problem: CarProblem,
    evidence: list[Evidence],
) -> tuple[list[Cause], list[PartSuggestion], list[DiagnosticStep], list[str]]:
    """Validate model output field by field, dropping what does not hold up.

    A cause citing evidence index 99 when only 12 items exist is a fabricated
    citation; it gets dropped and recorded rather than silently rendered.
    """
    warnings: list[str] = []
    n = len(evidence)

    causes: list[Cause] = []
    for raw in data.get("causes") or []:
        ids = [i for i in (raw.get("evidence_ids") or []) if isinstance(i, int)]
        valid = [i for i in ids if 0 <= i < n]
        if len(valid) != len(ids):
            warnings.append(
                f"Cause {raw.get('title')!r} cited out-of-range evidence "
                f"{sorted(set(ids) - set(valid))}"
            )

        basis = raw.get("basis")
        if basis not in {"evidence", "standard_diagnosis"}:
            # Infer from whether anything actually resolved.
            basis = "evidence" if valid else "standard_diagnosis"
        if basis == "evidence" and not valid:
            # Claimed vehicle-specific backing but cited nothing that exists.
            # Keep the cause, demote the claim — the mechanic's differential is
            # still worth showing, the false provenance is not.
            basis = "standard_diagnosis"
            warnings.append(
                f"Cause {raw.get('title')!r} claimed evidence but cited none "
                "that resolved — reclassified as standard diagnosis."
            )

        confidence = max(0.0, min(float(raw.get("confidence", 0.3)), 1.0))
        if basis == "standard_diagnosis":
            confidence = min(confidence, 0.5)

        try:
            causes.append(
                Cause(
                    title=raw.get("title") or "unnamed",
                    explanation=raw.get("explanation") or "",
                    confidence=confidence,
                    basis=basis,
                    evidence_ids=valid if basis == "evidence" else [],
                    typical_symptoms=raw.get("typical_symptoms") or [],
                    severity=raw.get("severity") if raw.get("severity") in
                    {"low", "medium", "high", "safety-critical"} else "medium",
                )
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Malformed cause dropped: {exc}")

    causes.sort(key=lambda c: c.confidence, reverse=True)

    parts: list[PartSuggestion] = []
    for raw in data.get("parts") or []:
        kind = raw.get("oem_or_aftermarket")
        parts.append(
            PartSuggestion(
                name=raw.get("name") or "unnamed part",
                part_number=raw.get("part_number") or None,
                oem_or_aftermarket=kind if kind in {"oem", "aftermarket"} else "unknown",
                notes=raw.get("notes"),
                evidence_ids=[
                    i for i in (raw.get("evidence_ids") or [])
                    if isinstance(i, int) and 0 <= i < n
                ],
            )
        )

    steps = [
        DiagnosticStep(
            step=raw.get("step") or "",
            why=raw.get("why") or "",
            requires_shop=bool(raw.get("requires_shop")),
        )
        for raw in (data.get("next_steps") or [])
        if raw.get("step")
    ]

    return causes, parts, steps, warnings


# Shown in the chat while a tool runs. Phrased as what is being consulted, not
# as a function name — "Checking safety recalls" is legible, check_recalls is not.
_TOOL_NARRATION = {
    "lookup_obd_codes": "Looking up the fault code",
    "search_owner_complaints": "Checking what other owners reported to the regulator",
    "get_failure_priors": "Checking which parts fail most on this model",
    "check_recalls": "Checking for open safety recalls",
    "search_web": "Searching Armenian and regional parts sites",
    "find_services": "Looking for a workshop in Armenia",
    "scrape_page": "Opening a listing for price and fitment",
}


def research_stream(
    raw_text: str,
    vehicle_hint: VehicleContext | None = None,
    max_rounds: int | None = None,
    image_findings: list[str] | None = None,
) -> Iterator[ResearchEvent]:
    """Run a research pass, yielding progress as it goes.

    Same pipeline as `research()`, surfaced incrementally so a chat UI can show
    the reasoning as it happens. The final event is kind="report" and carries
    the complete ResearchReport.
    """
    started = time.monotonic()
    meter = UsageMeter()
    warnings: list[str] = []

    try:
        client = ReasoningClient(meter=meter)
    except LLMError as exc:
        yield ResearchEvent(kind="error", message=str(exc))
        return

    caps = config.Capabilities.detect()
    if not (caps.perplexity_fast or caps.firecrawl):
        warnings.append(
            "No search provider configured — report is built from NHTSA, OBD "
            "reference and vPIC only."
        )
    elif not caps.perplexity_fast:
        warnings.append(
            "Web search running on Firecrawl; sonar-deep-research is unavailable."
        )

    # --- intake ---------------------------------------------------------
    yield ResearchEvent(kind="intake", message="Understanding the problem…")
    try:
        problem = intake(raw_text, client, hint=vehicle_hint)
    except LLMError as exc:
        yield ResearchEvent(kind="error", message=f"Could not read the problem: {exc}")
        return
    if image_findings:
        problem.image_findings = image_findings

    # Cross-check the vehicle against the official catalogue. Advisory only —
    # plenty of cars on Armenian roads were never sold in the US market that
    # vPIC indexes, so a miss means "unconfirmed", not "wrong".
    warnings.extend(vpic.check_vehicle(problem.vehicle))

    yield ResearchEvent(
        kind="intake",
        message=(
            f"{problem.vehicle.label()} — "
            + (", ".join(problem.symptoms) or "symptoms unclear")
            + (f" (codes: {', '.join(problem.obd_codes)})" if problem.obd_codes else "")
        ),
        data={"problem": problem.model_dump(mode="json")},
    )

    # --- research loop --------------------------------------------------
    dispatcher = build_dispatcher(problem.vehicle)
    schemas = available_tool_schemas(problem.vehicle, problem.obd_codes)
    photo_block = (
        f"Findings from user photos: {'; '.join(problem.image_findings)}\n"
        if problem.image_findings else ""
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": prompts.RESEARCH.format(
                vehicle=problem.vehicle.label(),
                problem=problem.raw_text,
                symptoms=", ".join(problem.symptoms) or "none stated",
                codes=", ".join(problem.obd_codes) or "none",
                photo_block=photo_block,
                source_block=prompts.source_block(
                    [f"{s['function']['name']}: {s['function']['description']}"
                     for s in schemas]
                ),
            ),
        },
        {"role": "user", "content": "Begin gathering evidence."},
    ]

    evidence: list[Evidence] = []
    call_log: list[str] = []
    rounds = 0
    cap = max_rounds or config.MAX_TOOL_ROUNDS

    for rounds in range(1, cap + 1):
        yield ResearchEvent(
            kind="round",
            message=f"Research round {rounds} of {cap}",
            data={"round": rounds, "evidence_so_far": len(evidence)},
        )
        try:
            msg = client.chat(messages, tools=schemas, max_tokens=1200)
        except LLMError as exc:
            warnings.append(f"Research loop stopped early: {exc}")
            break

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            name = tc.function.name
            call_log.append(name)
            yield ResearchEvent(
                kind="tool_start",
                message=_TOOL_NARRATION.get(name, f"Consulting {name}"),
                data={"tool": name, "arguments": tc.function.arguments},
            )

            found = execute_tool_call(name, tc.function.arguments, dispatcher)
            found = found[: max(config.MAX_EVIDENCE_ITEMS - len(evidence), 0)]
            offset = len(evidence)
            evidence.extend(found)

            yield ResearchEvent(
                kind="tool_result",
                message=(
                    f"Found {len(found)} result(s)" if found
                    else "Nothing found from that source"
                ),
                data={
                    "tool": name,
                    "count": len(found),
                    "items": [
                        {"index": offset + i, "title": e.title, "url": e.url,
                         "source": e.source_key}
                        for i, e in enumerate(found)
                    ],
                },
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": summarise_for_model(found, offset),
            })

        if len(evidence) >= config.MAX_EVIDENCE_ITEMS:
            break

    evidence = _dedupe(evidence)
    for failure in sorted(PROVIDER_FAILURES):
        warnings.append(
            f"SOURCE UNAVAILABLE — {failure}. Results below are missing that "
            "source entirely; this is not the same as it finding nothing."
        )
    PROVIDER_FAILURES.clear()
    if not evidence:
        warnings.append("No evidence retrieved — every source returned empty.")

    # --- synthesis ------------------------------------------------------
    yield ResearchEvent(
        kind="synthesis",
        message=f"Weighing {len(evidence)} pieces of evidence…",
        data={"evidence_count": len(evidence)},
    )
    try:
        data = _synthesise(problem, evidence, client)
    except LLMError as exc:
        yield ResearchEvent(kind="error", message=f"Could not write the report: {exc}")
        return

    causes, parts, steps, synth_warnings = _coerce_report(data, problem, evidence)
    warnings.extend(synth_warnings)
    warnings.extend(_verify_part_claims(parts, evidence, problem.vehicle))

    report = ResearchReport(
        problem=problem,
        summary=data.get("summary") or
        "Could not produce a grounded diagnosis from the available sources.",
        causes=causes,
        parts=parts,
        next_steps=steps,
        evidence=evidence,
        rounds_used=rounds,
        tool_calls=call_log,
        tokens_in=meter.tokens_in,
        tokens_out=meter.tokens_out,
        est_cost_usd=round(meter.est_cost_usd, 5),
        duration_s=round(time.monotonic() - started, 2),
        warnings=warnings,
        cache_path="deep_research",
    )

    yield ResearchEvent(
        kind="report",
        message=report.summary,
        data={"report": report.model_dump(mode="json"),
              "citations": [c.model_dump(mode="json") for c in report.citations()]},
    )


def research(
    raw_text: str,
    vehicle_hint: VehicleContext | None = None,
    max_rounds: int | None = None,
    image_findings: list[str] | None = None,
) -> ResearchReport:
    """Blocking wrapper over `research_stream`. Returns the finished report."""
    final: ResearchReport | None = None
    error: str | None = None
    for event in research_stream(raw_text, vehicle_hint, max_rounds, image_findings):
        if event.kind == "report":
            final = ResearchReport.model_validate(event.data["report"])
        elif event.kind == "error":
            error = event.message
    if final is None:
        raise LLMError(error or "Research produced no report.")
    return final


__all__ = ["research", "research_stream", "intake", "LLMError"]
