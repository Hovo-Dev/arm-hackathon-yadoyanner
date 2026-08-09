# Deep research component

Pure-Python PoC. No Django yet — `research.agent.research()` is the single
entry point Django will call later.

```bash
pip install -r requirements.txt
python scripts/ingest_obd.py          # optional: bulk code list for breadth
python run_research.py --status       # what is wired up right now
python run_research.py "Nissan Altima 2015, P0420, smells of fuel at idle"
```

## Pipeline

```
intake      free text (hy/ru/en) -> CarProblem      [1 LLM call]
research    bounded tool-calling loop               [<= MAX_TOOL_ROUNDS calls]
            each tool returns Evidence with provenance
dedupe      collapse repeat retrievals
synthesise  Evidence -> ranked causes + parts       [1 LLM call]
validate    drop any cause whose citations don't resolve
```

Typical run: ~30 evidence items, 6 rounds, ~90–120 s, **~$0.02**.

## Files

| File | Role |
|---|---|
| `config.py` | Keys, models, budget caps, source registry, blocked sources |
| `prompts.py` | **Every prompt in the system.** Each states the JSON it must return |
| `schemas.py` | Typed contracts. `ResearchReport` is the downstream integration surface |
| `llm.py` | DeepSeek transport, retries, JSON coercion, cost metering |
| `tools.py` | Tool schemas + dispatcher (the source allowlist boundary) |
| `agent.py` | The research loop and output validation |
| `sources/` | One adapter per source; all return `Evidence` |

Prompts live only in `prompts.py`. If you need a multi-line instruction
string anywhere else, it belongs there instead.

## What the model can reach

The agent cannot roam the open web. It reaches exactly what `tools.py`
dispatches, and web search is confined to `data/search_domains.json`. An
allowlist is a boundary; a prompt instruction is only a request.

Tools with no API key are **removed from the schema list** rather than left to
fail — otherwise the model burns paid rounds retrying them with reworded
queries.

| Source | Key needed | Notes |
|---|---|---|
| NHTSA complaints/recalls | none | Real symptom → dealer-diagnosis pairs. The "experienced mechanic" signal |
| OBD codes | none | Offline. See the warning below |
| Local manuals | none | `data/manuals/<make>/<model>/`. Owner's manuals only |
| Perplexity search | `PERPLEXITY_API_KEY` | OpenAI-compatible, goes through the same SDK |
| Firecrawl scrape | `FIRECRAWL_API_KEY` | Plain httpx — no OpenAI-shaped API |

## Two things that will bite you

**1. The public OBD dataset is wrong.** `mytrile/obd-trouble-codes` — mirrored
verbatim inside `aws-solutions/aws-connected-vehicle-solution` — is misaligned
across several ranges. 11 of 21 spot-checked common codes were incorrect:
P0420 → "Secondary Air Injection Relay B", P0300 → "Cylinder 12 Contribution",
P0700 → "Fuel Level Output Circuit". Wide adoption is not evidence of
correctness.

So `sources/obd.py` puts its hand-verified table **first** and the bulk list
second, flagged unverified and scored low. Before trusting any replacement
dataset:

```bash
python scripts/validate_obd.py --file candidate.json
```

**2. DeepSeek cannot see images.** Every DeepSeek model on OpenRouter is
`input_modalities: ["text"]`, and the hackathon key is provider-locked to
DeepSeek, so free vision models are unreachable too. Photos must be captioned
by a separate vision model upstream and passed in as text:

```bash
python run_research.py --photo-note "cracked rubber hose, oil residue" "..."
```

`CarProblem.image_findings` is where those land.

## Cost control

The key is $120 one-time. Every run is bounded by `MAX_TOOL_ROUNDS` (6) and
`MAX_EVIDENCE_ITEMS` (40); each report carries its own token count and cost
estimate. Check remaining budget:

```bash
curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" | python3 -m json.tool
```

## Grounding

`ResearchReport.grounded()` is the headline metric: true only when every stated
cause cites evidence indices that actually resolve. Causes citing out-of-range
indices are dropped before rendering and recorded in `warnings` — a fabricated
citation never reaches the user.

## Next

- KB layer: pgvector + a multilingual embedding model (`bge-m3` handles
  Armenian and Russian; `text-embedding-3-small` is weak on both) with the
  cache-hit / cache-miss split
- Seed the KB from NHTSA complaints so it is not cold on day one
- Eval harness over a fixed question set, scoring grounding rate and
  tool-selection accuracy, in the style of `scripts/validate_obd.py`
