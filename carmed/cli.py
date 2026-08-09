"""CLI for exercising the agentic layer with the dummy adapters.

    python -m carmed.cli "clicking noise when I turn left" --make Toyota --model Camry --year 2011
    python -m carmed.cli "brakes squeal" --make Toyota --model Camry --year 2012 --safety-floor
    python -m carmed.cli "clicking noise" --vin 1FTFW1ET5DFC10312     # refuses: bad checksum
    python -m carmed.cli "clicking noise" --offline                   # no model, no spend
    python -m carmed.cli --graph                                      # print the graph

Needs OPENROUTER_API_KEY unless --offline.
"""

from __future__ import annotations

import argparse
import os
import sys

from carmed import Answer, AnswerStatus, Query, Vehicle, run

MODEL = "deepseek/deepseek-v4-pro"
BASE_URL = "https://openrouter.ai/api/v1"


def make_watcher():
    """Prints what is actually sent to and returned by the model.

    LangChain's own ``set_debug`` prints a tree of internal chain names with
    ``[inputs]`` placeholders where the content should be, which tells you
    nothing. This prints the two things that matter: the last message sent,
    and the reply. Tool calls are already visible in ``--trace``.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    def clip(text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[:limit] + " …"

    class Watcher(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, **kwargs):
            turn = messages[0]
            print(f"\n\033[36m→ MODEL CALL\033[0m  ({len(turn)} messages in context)")
            for message in turn[-2:]:
                role = type(message).__name__.replace("Message", "")
                if getattr(message, "tool_calls", None):
                    for call in message.tool_calls:
                        print(f"   [{role}] wants {call['name']}({call['args']})")
                elif message.content:
                    print(f"   [{role}] {clip(message.content, 400)}")

        def on_llm_end(self, response, **kwargs):
            gen = response.generations[0][0]
            calls = getattr(gen.message, "tool_calls", None) if hasattr(gen, "message") else None
            if calls:
                for call in calls:
                    print(f"\033[33m← CALLS TOOL\033[0m {call['name']}({call['args']})")
            if gen.text.strip():
                print(f"\033[32m← REPLY\033[0m {clip(gen.text, 600)}")

    return Watcher()


def build_model(watch: bool = False):
    from langchain_openai import ChatOpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set. Use --offline to run without a model.")
    return ChatOpenAI(
        model=MODEL, api_key=key, base_url=BASE_URL,
        temperature=0.2, timeout=90, max_retries=1,
        callbacks=[make_watcher()] if watch else None,
    )


def render(answer: Answer) -> str:
    """Turn an Answer into terminal text.

    Ordered by what a worried car owner needs first: what car we think this
    is, then how urgent it is, then what is wrong, then what to buy, then who
    to call. The three non-answered statuses each return early, because a
    refusal or a question should not be buried under empty sections.
    """
    out: list[str] = [f"CAR       {answer.vehicle.describe()}"]
    v = answer.vehicle
    if v.vin:
        flag = "checksum ok" if v.vin_valid else "CHECKSUM FAILED"
        out.append(f"VIN       {v.vin}  [{flag}]  built for {v.region.value}")
    out.append(f"INTENT    {answer.intent}")

    if answer.status is AnswerStatus.REFUSED_BAD_VIN:
        out += [
            "",
            f"REFUSED   {answer.message}",
            "",
            "Stopping rather than guessing: a wrong VIN sends you to buy an",
            "expensive part for a car you do not own.",
        ]
        return "\n".join(out)

    if answer.status is AnswerStatus.NEEDS_CLARIFICATION:
        return "\n".join(out + ["", f"QUESTION  {answer.message}"])

    out += ["", f"URGENCY   {answer.urgency.label.upper()}"]
    if answer.urgency_reason:
        out.append(f"          {answer.urgency_reason}")

    out.append("")
    if answer.status is AnswerStatus.ABSTAINED:
        out += ["LIKELY CAUSES", f"  {answer.message}"]
    elif answer.causes:
        out.append("LIKELY CAUSES")
        for i, cause in enumerate(answer.causes, 1):
            out.append(f"  {i}. {cause.title}   [{cause.confidence}]")
            if cause.explanation:
                out.append(f"     {cause.explanation}")
            if cause.evidence:
                out.append(f"     evidence: {', '.join(cause.evidence)}")

    if answer.repair_steps:
        out += ["", "REPAIR STEPS"]
        out += [f"  - {s}" for s in answer.repair_steps]

    for warning in answer.parts.fitment_warnings:
        out += ["", f"FITMENT [{warning.severity}]", f"  {warning.message}"]

    # Prices and URLs come from `answer.listings`, never from the option --
    # this is the last point where a generated price could have slipped in,
    # and it cannot, because PartOption has no price field to carry one.
    if answer.parts.options:
        out += ["", "WHAT TO ORDER"]
        for option in answer.parts.options:
            out.append(f"  {option.name_en}")
            if option.aliases:
                out.append(f"    also listed as: {', '.join(option.aliases[:4])}")
            for lid in option.listing_ids[:5]:
                listing = answer.listings.get(lid)
                if not listing:
                    continue
                # Missing price is normal on list.am -- sellers often omit it
                # and expect a phone call. Say so rather than printing 0.
                price = str(listing.price) if listing.price else "no price"
                flag = "   <-- suspiciously cheap" if lid in option.suspicious_listing_ids else ""
                out.append(f"    - {listing.title[:60]}")
                out.append(f"      {price} | {listing.condition or '?'} | {listing.url}{flag}")

    if answer.shops:
        out += ["", "WHOM TO CALL"]
        out += [f"  {s.name} | {s.phone} | {s.area or ''}" for s in answer.shops]

    out += [
        "",
        f"from_cache={answer.from_cache}  dropped_refs={answer.dropped_refs}",
        "",
        answer.disclaimer,
    ]
    return "\n".join(out)


def render_trace(trace) -> str:
    """What ran, what it cost, what it fetched."""
    out = [
        "",
        "─" * 62,
        "TRACE",
        "─" * 62,
        f"  steps      {' -> '.join(trace.steps)}",
        f"  llm calls  {', '.join(trace.llm_calls) or 'none'}   (cost: {trace.cost})",
        "  lookups    " + ("\n             ".join(trace.lookups) or "none"),
        "  notes",
    ]
    out += [f"      {n}" for n in trace.notes]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="carmed", description="Car maintenance assistant")
    p.add_argument("text", nargs="?", default="", help="the problem, in any language")
    p.add_argument("--vin")
    p.add_argument("--make")
    p.add_argument("--model", dest="car_model")
    p.add_argument("--year", type=int)
    p.add_argument("--engine", type=float, help="displacement in litres")
    p.add_argument("--city", default="Yerevan")
    p.add_argument("--mileage", type=int)
    p.add_argument("--offline", action="store_true", help="run with no model")
    p.add_argument("--safety-floor", action="store_true",
                   help="force 'do not drive' for brakes/steering/suspension/airbags")
    p.add_argument("--trace", action="store_true",
                   help="show steps, model calls and lookups")
    p.add_argument("--verbose", action="store_true",
                   help="show each prompt sent and each reply received")
    p.add_argument("--json", action="store_true", help="print the Answer as JSON")
    p.add_argument("--graph", action="store_true", help="print the graph and exit")
    p.add_argument("--dummy", action="store_true",
                   help="use the fake corpus instead of live Armenian sources")
    args = p.parse_args(argv)

    from adapters.dummy import DummyCaseStore, DummyResearch

    # The single line a real deployment changes: swap these two objects for
    # your implementations. Nothing else in the CLI knows the difference.
    #
    # ResearchTool is live: list.am and Russian suppliers for parts, turn.am's
    # directory for workshops, NHTSA for complaints. CaseStore is still the
    # dummy — the pgvector store lives in the Django app, not here.
    if args.dummy or args.graph:
        research = DummyResearch()
    else:
        from adapters.armenian import ArmenianResearch

        research = ArmenianResearch()
    case_store = DummyCaseStore()

    # --graph needs a compiled graph but no model and no real request, so the
    # dummy adapters are enough and nothing is spent.
    if args.graph:
        from carmed.graph import build

        graph, _ = build(research=research, case_store=case_store)
        print(graph.get_graph().draw_mermaid())
        return 0

    if not args.text:
        p.error("give me a problem description, or use --graph")

    answer = run(
        Query(
            text=args.text,
            vehicle=Vehicle(
                make=args.make, model=args.car_model, year=args.year,
                engine_l=args.engine, vin=args.vin,
            ),
            city=args.city,
            mileage_km=args.mileage,
        ),
        research=research,
        case_store=case_store,
        model=None if args.offline else build_model(watch=args.verbose),
        safety_floor=args.safety_floor,
    )

    if args.json:
        print(answer.model_dump_json(indent=2))
    else:
        print(render(answer))
        if args.trace:
            print(render_trace(answer.trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
