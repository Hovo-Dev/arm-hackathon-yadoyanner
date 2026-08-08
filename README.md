# carmed — agentic layer

Diagnosis, parts and shops for cars in Armenia. You give it a sentence in
Armenian, Russian or English; it returns probable causes, an urgency call,
what to order with real prices and links, and who to call.

This package is **only the agentic layer**. It owns no database, no embedding
model and no HTTP client. It asks for those through two Protocols that your
Django layer implements.

```
Query ──> [VIN check] ──> [intent router] ──> [case gate] ──> [diagnostician]
                                                    │                │
                                                    │          [parts explorer]
                                                    │                │
                                                    └──────> [shops] ──> Answer
```

---

## 1. Install

```bash
pip install -r requirements.txt
```

Python 3.12+ (developed on 3.14). Runtime deps: `langgraph`, `langchain-openai`,
`pydantic`.

## 2. Run the CLI

```bash
export OPENROUTER_API_KEY="sk-or-..."
python -m carmed.cli "squealing sound when I press the brake pedal in the morning" --make Toyota --model Camry --year 2012
```

Useful flags:

```bash
python -m carmed.cli "clicking when I turn left" --make Toyota --model Camry --year 2011 --trace
python -m carmed.cli "arm brakes" --vin 1FTFW1ET5DFC10312    # refuses: bad checksum
python -m carmed.cli "clicking noise" --offline               # no model, no spend
python -m carmed.cli "brakes squeal" --safety-floor           # force "do not drive"
python -m carmed.cli "clicking noise" --json                  # Answer as JSON
python -m carmed.cli --graph                                  # mermaid diagram
```

The CLI runs against `adapters/dummy.py`, which fakes both ports with a small
hardcoded corpus. Everything works end to end before your real research tool
exists.

```bash
python tests/test_carmed.py     # 27 tests, offline, no budget
```

---

## 3. Input and output

**Input — `carmed.Query`**

| field | type | notes |
|---|---|---|
| `text` | `str` | the problem, any language |
| `vehicle` | `Vehicle` | `make`, `model`, `year`, `engine_l`, `vin` — all optional |
| `city` | `str` | defaults to `"Yerevan"` |
| `mileage_km` | `int \| None` | |

**Output — `carmed.Answer`**

| field | type | notes |
|---|---|---|
| `status` | `answered` / `abstained` / `needs_clarification` / `refused_bad_vin` | |
| `vehicle` | `Vehicle` | with `vin_valid` and `region` filled in |
| `intent` | `Intent` | what the router decided |
| `causes` | `list[Cause]` | ≤3, most likely first, each with `evidence` ids |
| `urgency` | `Urgency` | `0` drive it → `3` do not drive. `.label` for text |
| `repair_steps` | `list[str]` | |
| `parts` | `PartsResult` | `options` + `fitment_warnings` |
| `shops` | `list[Shop]` | |
| `listings` | `dict[str, PartListing]` | resolve `option.listing_ids` against this |
| `message` | `str` | set when abstained / refused / needs clarification |
| `from_cache` | `bool` | true when the case gate answered it |
| `dropped_refs` | `int` | unresolvable references removed. **Should be 0** |
| `trace` | `Trace` | what ran, what it cost, what it fetched — see below |

Both are pydantic, so `answer.model_dump(mode="json")` goes straight into a
serializer.

```python
from carmed import Query, Vehicle, run

answer = run(
    Query(text="clicking when I turn left",
          vehicle=Vehicle(make="Toyota", model="Camry", year=2011)),
    research=my_research_tool,
    case_store=my_case_store,
    model=chat_model,          # any LangChain chat model; None = no model
    safety_floor=False,
)
```

### Verifying what ran — `Answer.trace`

Every request records itself. `answer.trace` is a `Trace` with four lists:

| field | what it answers |
|---|---|
| `steps` | which steps executed, in order |
| `llm_calls` | which agents made a model call — **the length is the bill** |
| `lookups` | every call out to your ports, as `name->result_count` |
| `notes` | decisions, warnings, failures |

Plus `trace.cost` (number of model calls) and three helpers: `trace.ran("gate")`,
`trace.called("diagnostician")`, `trace.looked_up("case_store.find_similar")`.

```bash
python -m carmed.cli "squealing when I press the brake pedal" --make Toyota --model Camry --year 2012 --trace
```

```
  steps      vehicle -> route -> gate -> diagnose -> finalize
  llm calls  router, diagnostician   (cost: 2)
  lookups    case_store.find_similar->1
             research.search_knowledge->3
             research.search_knowledge->1
  notes
      car=2012 Toyota Camry region=?? lang=en area=brakes
      intent=diagnose (router)
      cache miss (top=0.23 tier=exact)
          searched knowledge for 'squealing sound when pressing brake pedal...' -> 3
          searched knowledge for 'brake squeal cold morning moisture...' -> 1
      status=needs_clarification
```

Read it top to bottom: `parts` and `shops` never ran because the diagnostician
asked a clarifying question, which exits early. Two model calls. The
diagnostician searched twice, rewording after the first pass came back thin.

This is what the routing tests assert against — see `tests/test_carmed.py`,
which pins the exact step sequence for each kind of question so a routing
change cannot pass silently.

> **A generated price can never reach the answer.** Agents *do* see prices —
> they need them to flag suspiciously cheap listings — but there is nowhere for
> a generated one to land: `PartOption` carries `listing_ids` and no price
> field, and every displayed price and URL is rendered from the record your
> tool returned. Ids that resolve to nothing are dropped and counted in
> `dropped_refs`; free text is scanned by `carmed.text.find_leaks` for anything
> money-, phone- or URL-shaped. Shop phone numbers never pass through a model
> at all.

---

## 4. Plugging in your tools

You implement two things. Neither needs a base class or registration — if the
methods match, it works.

```python
from carmed import run, Query, Vehicle

answer = run(
    Query(text="brakes squeal", vehicle=Vehicle(make="Toyota", year=2012)),
    research=MyResearchTool(),     # <- yours
    case_store=MyCaseStore(),      # <- yours
    model=my_chat_model,
)
```

That is the entire wiring. There is no config file and no plugin registry.

---

### 4.1 `ResearchTool` — searching the world

Copy this skeleton and fill in the three methods.

```python
from carmed import (
    KnowledgeSnippet, PartListing, Shop, Money, SpecRegion, SystemArea, Vehicle,
)

class MyResearchTool:

    def search_knowledge(self, *, query, vehicle, limit=6) -> list[KnowledgeSnippet]:
        """Manuals, catalogs, forum threads. Called by the diagnostician."""
        rows = my_search(query, make=vehicle.make, model=vehicle.model)
        return [
            KnowledgeSnippet(
                id=row.stable_id,          # MUST be stable across calls
                text=row.body,             # ~600 chars is plenty
                source_url=row.url,        # MUST be real
                title=row.title,
                kind="forum",              # forum | manual | catalog
            )
            for row in rows[:limit]
        ]

    def search_listings(self, *, part_terms, vehicle, limit=20) -> list[PartListing]:
        """Parts for sale. Called by the parts explorer.

        `part_terms` arrives multilingual on purpose, e.g.
        ["wheel bearing", "подшипник ступицы", "անվահեծի առանցքակալ"].
        Search for ALL of them and merge — each spelling finds different rows.
        """
        rows = my_listing_search(part_terms, make=vehicle.make, year=vehicle.year)
        return [
            PartListing(
                id=row.id,
                title=row.title,           # keep the seller's original wording
                url=row.url,
                price=Money(amount=row.price_amd, currency="AMD"),
                condition=row.condition,   # new | used | dismantler
                region=SpecRegion.EU if row.is_euro_part else SpecRegion.UNKNOWN,
            )
            for row in rows[:limit]
        ]

    def search_shops(self, *, make, system_area, city, limit=5) -> list[Shop]:
        """Repair shops. No model touches this — what you return is shown."""
        rows = Shop.objects.filter(city=city)   # or wherever these live
        return [
            Shop(id=str(s.pk), name=s.name, phone=s.phone, area=s.district,
                 specialties=[SystemArea(x) for x in s.specialties])
            for s in rows[:limit]
        ]
```

**The rules, in order of how badly breaking them hurts:**

1. **`id` must be stable across calls.** Agents cite `listing:L4`; the answer
   renders the price and URL from the record with that id. An id that changes
   between two searches in the same request means the citation resolves to
   nothing and gets silently dropped (you will see it in
   `answer.dropped_refs`). Use your primary key, not a row number.
2. **`url` / `source_url` must be real.** They are shown to the user and are
   what satisfies list.am's `Content-Signal: use=reference`. Never synthesise
   one.
3. **Return `[]` freely.** Every step degrades: no listings means no parts
   block, no knowledge means the diagnostician works from stored cases alone.
   The model is told to reword and retry on an empty result. Fabricating a
   row to "be helpful" is the one unrecoverable mistake.
4. **Honour `limit`.** Anything past 25 records is truncated before it reaches
   the model anyway (`MAX_RENDERED` in `carmed/agents.py`), so returning 500
   rows just wastes your own time. All returned records *are* kept for
   citation resolution, so nothing you return is lost — it just may not be
   seen.
5. **Keep the seller's original title.** Do not normalise or translate it —
   grouping the three spellings is the parts agent's job, and it needs the
   raw text to do it.
6. **Set `region` when you can tell.** It drives the US-vs-Euro fitment
   warning. Leave it `UNKNOWN` and the fallback only catches obvious markers
   like `EURO SPEC` in the title.

**Failures are handled for you.** If a method raises, the agent is told
"The parts search is unavailable right now" and can try different wording;
the real exception goes to `answer.trace.notes`, never into the prompt.

**Expect more calls than you think.** The prompt asks for two searches; models
have been observed doing five. The hard bound is `RECURSION_LIMIT = 8`. Budget
for up to ~6 real searches per request and cache aggressively — the same
query often repeats within one request.

---

### 4.2 `CaseStore` — searching your own past cases

One method. This is the case gate, and it is **deliberately not an agent** —
embedding a sentence and ranking rows by distance is a database query, and a
model in front of it would add cost and latency without adding judgement.

```python
from carmed import Case, CaseMatch, MatchTier, Vehicle

class MyCaseStore:
    def find_similar(self, *, text, vehicle, limit=5) -> list[CaseMatch]:
        # 1. Filter by vehicle FIRST, so a Camry question cannot match a BMW.
        qs = MaintenanceCase.objects.filter(retrievable=True, split="kb")
        tier = MatchTier.LOOSE
        if vehicle.make:
            qs = qs.filter(make__iexact=vehicle.make)
        if vehicle.make and vehicle.model:
            qs = qs.filter(model__iexact=vehicle.model)
            tier = MatchTier.NEAR
            if vehicle.year:
                tight = qs.filter(year__range=(vehicle.year - 1, vehicle.year + 1))
                if tight.exists():
                    qs, tier = tight, MatchTier.EXACT

        # 2. Only now rank by similarity.
        rows = (qs.annotate(distance=CosineDistance("embedding", embed(text)))
                  .order_by("distance")[:limit])

        return [
            CaseMatch(
                case=Case(id=str(r.pk), symptom=r.symptom, fix=r.fix, ...),
                score=1 - float(r.distance),   # 0..1, higher is closer
                tier=tier,
            )
            for r in rows
        ]
```

**`tier` is the field people get wrong.** It says how tightly *you* filtered,
and it gates what the answer is allowed to do:

| tier | meaning | may serve a cached answer? |
|---|---|---|
| `EXACT` | make + model + year ±1 (+ engine) | yes |
| `NEAR` | make + model, nearby years | yes |
| `LOOSE` | make only, or symptom only | **no — context only** |

A `LOOSE` match still reaches the diagnostician as evidence. It just cannot
*be* the answer: "some Toyota had this once" is not an answer about this car.

**A case is only served from cache when all four hold:** `verified` is true,
`tier` is EXACT or NEAR, `score >= CACHE_HIT_THRESHOLD` (0.80), and it is the
top match. Otherwise everything falls through to the diagnostician.

> **Recalibrate `CACHE_HIT_THRESHOLD` whenever you change embedding model.**
> Cosine similarity has no absolute meaning across models — 0.80 on one is
> 0.62 on another. Calibrate on held-out examples, never on your evaluation
> set.

**Never return held-out or unverified rows.** Enforce it in the queryset or a
DB constraint, not by remembering to filter at the call site. Note that
`find_similar` has no `include_eval` argument on purpose: there is no way for
a caller to ask for them.

---

### 4.3 Checking your implementation works

Swap your objects into the CLI and watch the trace:

```bash
python -m carmed.cli "brakes squeal in the morning" --make Toyota --model Camry --year 2012 --trace
```

Look for these lines:

```
lookups    case_store.find_similar->3      <- your store was called, 3 rows back
           research.search_knowledge->6    <- your search ran
           research.search_listings->12
```

Then check the three things that indicate a broken implementation:

| symptom | cause |
|---|---|
| `dropped_refs` > 0 | your ids are not stable between calls |
| `case_store.find_similar->0` every time | vehicle filter too tight, or embeddings not populated |
| cache never hits | `tier` always LOOSE, or `verified` not set, or threshold miscalibrated |

`answer.trace` is a pydantic model, so assert against it in your own tests:

```python
t = run(query, research=MyResearch(), case_store=MyStore()).trace
assert t.looked_up("research.search_listings")
assert t.cost <= 3
assert answer.dropped_refs == 0
```

## 5. Django integration

### Models

`carmed.Case` maps to a table you already need. Suggested columns:

```python
class MaintenanceCase(models.Model):
    class Provenance(models.TextChoices):
        FORUM_CONFIRMED = "forum_confirmed"
        MECHANIC_LABELED = "mechanic_labeled"
        LLM_GENERATED = "llm_generated"

    class Split(models.TextChoices):
        KB = "kb"; DEV = "dev"; EVAL = "eval"

    symptom      = models.TextField()
    fix          = models.TextField()
    make         = models.CharField(max_length=64, null=True, db_index=True)
    model        = models.CharField(max_length=64, null=True)
    year         = models.SmallIntegerField(null=True)
    engine_l     = models.DecimalField(max_digits=3, decimal_places=1, null=True)
    system_area  = models.CharField(max_length=16, default="unknown", db_index=True)
    part_names   = ArrayField(models.TextField(), default=list, blank=True)
    urgency      = models.SmallIntegerField(default=1)
    source_url   = models.URLField(null=True)

    provenance   = models.CharField(max_length=20, choices=Provenance.choices)
    split        = models.CharField(max_length=4, choices=Split.choices, default=Split.KB)
    retrievable  = models.BooleanField(default=True)
    embedding    = VectorField(dimensions=1024, null=True)   # pgvector

    class Meta:
        constraints = [
            # Held-out rows can never be retrieved. Enforced by the DB, not by
            # remembering to filter.
            models.CheckConstraint(
                check=~models.Q(split="eval") | models.Q(retrievable=False),
                name="eval_never_retrievable",
            )
        ]
```

Three separate fields, not one flag: `provenance` is *how much we trust this*,
`split` is *may it be retrieved at all*, `retrievable` is the enforced switch.
Write LLM-generated cases back with `retrievable=False` and promote by hand —
a generated case that can be retrieved as a confirmed fix lets the system
reinforce its own guesses.

**Filter before you rank.** With a global HNSW index Postgres applies `WHERE`
*after* the ANN traversal, so a selective vehicle filter can return almost
nothing and quietly destroy recall. Either pre-filter to a candidate set and do
exact cosine over it (fine up to a few thousand rows), or enable pgvector 0.8's
`hnsw.iterative_scan`.

### Adapter

```python
# services/case_store.py
from carmed import Case, CaseMatch, MatchTier, Vehicle

class DjangoCaseStore:
    def find_similar(self, *, text, vehicle, limit=5):
        qs = MaintenanceCase.objects.filter(retrievable=True, split="kb")
        tier = MatchTier.LOOSE
        if vehicle.make:
            qs = qs.filter(make__iexact=vehicle.make)
        if vehicle.make and vehicle.model:
            qs = qs.filter(model__iexact=vehicle.model)
            tier = MatchTier.NEAR
            if vehicle.year:
                exact = qs.filter(year__range=(vehicle.year - 1, vehicle.year + 1))
                if exact.exists():
                    qs, tier = exact, MatchTier.EXACT

        vector = embed(text)
        rows = (qs.annotate(distance=CosineDistance("embedding", vector))
                  .order_by("distance")[:limit])
        return [
            CaseMatch(case=to_case(row), score=1 - float(row.distance), tier=tier)
            for row in rows
        ]
```

### View

```python
# views.py
from carmed import Query, Vehicle, run

class DiagnoseView(APIView):
    def post(self, request):
        data = DiagnoseSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        answer = run(
            Query(
                text=data.validated_data["text"],
                vehicle=Vehicle(**data.validated_data.get("vehicle", {})),
                city=data.validated_data.get("city", "Yerevan"),
            ),
            research=DeepResearchTool(),
            case_store=DjangoCaseStore(),
            model=get_chat_model(),
        )
        return Response(answer.model_dump(mode="json"))
```

`run()` is synchronous and does network I/O — put it behind Celery, or accept
the request-thread block. Django's ORM is sync, so nothing here needs
`sync_to_async`.

---

## 6. Cost

Ceiling of **3 model calls** per request: router, diagnostician, parts
explorer. A case-gate hit removes the diagnostician. Intent routing removes
more — a shop lookup costs one call, not three.

Everything else is deterministic and free: VIN maths, language detection,
symptom routing, case matching, shop lookup, fitment checks, grounding.

The ReAct loop is capped at `RECURSION_LIMIT = 8` (`carmed/agents.py`), so a
confused model cannot spend the budget in a loop.

---

## 7. Things found the hard way

**DeepSeek rejects strict JSON schema.** `create_react_agent(response_format=…)`
uses OpenAI's `json_schema` mode, and this gateway answers
`This response_format type is unavailable now`. Probing the provider directly:
tool calling ✅, multi-turn tool round-trips ✅, `{"type": "json_object"}` ✅,
tools + JSON mode together ✅, `json_schema` ❌. So agents run in JSON mode with
the schema in the prompt and the final message is parsed in
`carmed/agents.py: invoke`.

Contrary to the access guide's warning, **DeepSeek's multi-step tool use works
fine** — it reliably searched twice with different wording when the first pass
was thin.

**A JSON schema alone is not enough.** Given only `model_json_schema()`, the
model invented its own field names for nested objects (`cause` instead of
`title`, and dropped `urgency`). Each agent's prompt now carries a worked
example (`carmed/agents.py: EXAMPLES`) — that fixed it. If you add a field,
update the example.

**The VIN checksum does not catch everything.** The transliteration table maps
several characters to the same number (1/A/J, 2/B/K/S, …), so a substitution
inside a class is invisible. It survives contact with reality because I, O and
Q are illegal in a VIN — killing the 0/O and 1/I confusions outright — and
every pair a camera actually confuses (5/S, 8/B, 2/Z, 6/G, 1/7) falls in a
different class. Claim "every realistic OCR error", not "every error".

---

## 8. The files

```
carmed/
  models.py    the data shapes
  ports.py     the two things you build
  vehicle.py   VIN checks and part-fitment warnings
  text.py      reading the user's words
  agents.py    everything to do with the AI
  graph.py     the conductor
  cli.py       terminal harness
  __init__.py  the front door
adapters/
  dummy.py     fakes, so it runs today
tests/
  test_carmed.py
```

Imports point one way. `models.py` is the base; everything else builds on it;
nothing in `carmed/` imports `adapters/`.

**`models.py`** — the nouns. Every piece of data that moves around: a question,
a car, a diagnosis, a part listing, an answer. Shapes only, no behaviour. Note
that `Diagnosis` and `PartsResult` double as the AI's output format, so
changing a field there changes a prompt.

**`ports.py`** — the two things you have to build. One searches your database
of past cases; one searches the web for parts, shops and manuals. This file
describes them and contains nothing else. Read it first.

**`vehicle.py`** — everything about the car itself. Checks whether a VIN is
real using its built-in checksum, works out which market the car was built for,
and warns when a part looks like it is for the wrong market. Pure functions,
no network.

**`text.py`** — reading the user's words. Works out whether they wrote
Armenian, Russian or English, and which system they mean (brakes, engine,
suspension). Also scans the finished answer for prices or phone numbers that
should not be there. No AI involved.

**`agents.py`** — everything to do with the AI. The instructions given to each
agent, the search tools they are allowed to call, worked examples of the reply
format, and the code that reads their answers back. `RECURSION_LIMIT` lives
here.

**`graph.py`** — the conductor. See below.

**`cli.py`** — lets you type a question in the terminal and see the answer.
The only shipped file that imports `adapters/`.

**`__init__.py`** — the front door. Django imports from here, never from the
submodules.

**`adapters/dummy.py`** — fake versions of your two pieces, with made-up data,
so the whole thing runs today. Delete it once the real ones exist; until then
it is a worked example of the shapes you must return.

### What `graph.py` does

It decides **what runs, in what order, and when to stop early** — and it is the
only file that knows about all the others.

Everything else is a pile of parts: `vehicle.py` can check a VIN, `agents.py`
can run an agent, your `CaseStore` can search. None of them know when they
should be used. `graph.py` is what turns them into one request.

It holds four things:

1. **`State`** — the clipboard passed from step to step. Starts with just the
   question and fills up as it goes: the car, the language, the intent, the
   matched cases, the diagnosis, the parts, the shops. Each step returns only
   the fields it changed.

2. **The steps** (`_resolve_vehicle`, `route`, `gate`, `diagnose`, `parts`,
   `shops`, `finalize`). Each is small, and mostly delegates: `_resolve_vehicle`
   calls into `vehicle.py`, `gate` calls your `CaseStore`, `diagnose` and
   `parts` call into `agents.py`. The real work lives elsewhere; these decide
   *whether* it happens and what to do with the result.

3. **The decisions** (`_after_vehicle`, `_after_route`, `_after_diagnose`,
   `_after_parts`). Four small functions that pick the next step by reading a
   value the previous one computed. A failed VIN checksum jumps straight to the
   end. Someone asking for a mechanic skips diagnosis and parts entirely.
   Someone whose answer is already in the database skips the AI. This is where
   the cost savings come from, and it is all plain `if` statements.

4. **`run()`** — the public entry point. Wires the steps together, runs them,
   and converts the finished clipboard into an `Answer`.

Two settings at the top are worth knowing: `CACHE_HIT_THRESHOLD` (how similar a
past case must be to reuse its answer) and `NEEDS` (which steps each kind of
question requires). Changing `NEEDS` changes the routing without touching any
other code.

It is the largest file, deliberately. Splitting the steps from the wiring would
mean reading two files to follow one request.

## 9. Not included, by design

VIN → build decoding (NHTSA vPIC is free and keyless:
`https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{VIN}?format=json`),
scraping, the evaluation harness, embeddings, persistence, and image/voice
input. `Vehicle.region` currently comes from the VIN's first character, which
is enough for the fitment check; add a decoder in the Django layer to fill in
make/model/year/engine automatically.
