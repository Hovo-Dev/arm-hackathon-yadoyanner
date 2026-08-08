# Car Maintenance Assistant — Armenia

**Hack Armenia · August 8–9, 2026**

One line: you describe a car problem in your own words, and the system tells you what is wrong, what part to order, where to buy it cheapest, and who to call.

---

## 1. Why this matters in Armenia

Armenia is not a normal car market. It is full of **rebuilt American crash cars**.

- Most vehicles came in through Georgia — damaged cars shipped from the USA, repaired inside Armenia with imported parts, then sold on or re-exported.
- The re-export business was worth roughly **USD 552m, about 3.7% of GDP** in the first eight months of 2023. The shadow economy is a major part of the sector.
- When the flow to Russia dropped, a large number of those cars **stayed in Armenia**.

This creates four problems that no existing site solves:

| Problem | What happens today |
|---|---|
| Car is US-spec, not Euro-spec | Owner orders the wrong part. Money gone. |
| No repair history | Nobody knows what was replaced or how badly it was fixed. |
| Parts market is chaotic | list.am, dismantlers, copies and originals, all mixed, in 3 languages. |
| No mechanic ratings | You find a mechanic through a cousin. |

And the stakes are real: the World Bank found Armenia has the **second-highest car death rate** among EU and former Soviet countries, costing around **6% of GDP**.

---

## 2. Input

Three ways in, all feeding the same engine:

- **Text** — "clicking noise when I turn left"
- **Voice** — same, spoken, in Armenian or Russian
- **Image** — see below

### Image use cases (ranked by how well they work)

1. **VIN plate photo → decode.** Highest value. Gives exact US-market build: model, trim, engine, plant, build date. Every later search is filtered by this.
2. **Photo of the old part.** The stamped numbers on it are the real key to finding a replacement.
3. **Dashboard warning light.** Fast, useful, low risk.
4. **Damage / leak photo.** Supporting signal only.

**Do not do:** diagnose from a photo of a part that makes a noise. You cannot see a bad wheel bearing.

---

## 3. Engine — agent layout

```
   [image]   [text]   [voice]
       \       |       /
        \      |      /
         v     v     v
    ┌──────────────────────────┐
    │  AGENT 4 — Case Gate     │   runs FIRST
    │  - metadata filter       │
    │    (brand, model, year)  │
    │  - embedding search      │
    │  - reads Case KB         │
    └──────────┬───────────────┘
               │
      match?   │   no match / low confidence
     ┌─────────┴─────────┐
     │                   │
     v                   v
 return aggregated   ┌───────────────────────────┐
 summary of prior    │ A1  Parts Explorer        │
 verified fixes      │ A2  Service Businesses    │
                     │ A3  Manual / Diag Expert  │
                     └────────────┬──────────────┘
                                  │
                                  v
                     ┌───────────────────────────┐
                     │  Deep Research Engine     │
                     │  list.am · auto.am        │
                     │  Autosan · car catalogs   │
                     │  car manuals · FORUMS     │
                     └────────────┬──────────────┘
                                  │
                                  v
                    SUMMARY: whom to call,
                    what to order, which site
```

### Agent roles

**Agent 4 — Case Gate (runs first, not last)**
Filters the Case Knowledge Base by car metadata, then does embedding search over prior solved cases. If a confident match exists, it returns an aggregated summary of fixes that worked and skips the expensive path. If not, it falls through to A1–A3.

> Note: the whiteboard labels this both "fallback" and "initial gate." Treat it as a **cache-first gate**. It runs first; A1–A3 are the fallback when the cache misses. This saves time and money on every repeat question, and repeat questions are most questions.

**Agent 1 — Parts Explorer**
Maps the diagnosis to a specific part for the exact VIN build. Searches list.am and parts sites. Groups duplicate listings written differently (`ամորտիզատոր` / `амортизатор` / `shock absorber` are one thing). Returns price spread, new vs used vs dismantler, and flags suspiciously cheap listings as likely copies.

**Agent 2 — Service Businesses**
Finds mechanics and shops that actually work on this brand and this job. Returns name, phone, area.

**Agent 3 — Manual / Diagnostic Expert**
Reads service manuals, catalogs and forum threads. Produces the diagnosis and the repair steps. Assigns urgency.

---

## 4. Output

A single summary with four blocks:

1. **What is probably wrong** — top 3 causes with confidence, not one confident answer
2. **Urgency** — `drive it` / `fix this week` / `do not drive`
3. **What to order** — part name, part number, price range, direct links, new vs used
4. **Whom to call** — shops, phone numbers, area

Every fact must trace to a source. No invented prices, no invented phone numbers.

---

## 5. Evaluation

### Ground truth: solved forum threads

Use forum threads where a user described a problem, received answers, and **confirmed the fix worked**. That confirmation is the label.

Build the set:
- Scrape threads with a confirmed-solution marker or an explicit "this fixed it" reply
- Keep: car make/model/year, symptom text, confirmed fix, parts named
- Target **150–200 threads**, minimum 100

### CRITICAL: prevent leakage

Forums are both a **data source** for the Deep Research Engine and the **ground truth** for evaluation. If the same threads are in both, your scores are meaningless and a judge will spot it.

**Split before you index anything:**

```
all scraped threads
      │
      ├── 70%  →  KB / retrieval index  (system may read these)
      └── 30%  →  EVAL SET  (excluded from index, never retrievable)
```

Enforce it in code — a hard thread-ID blocklist on the retriever, not a promise. Put this on a slide. Saying "we held out our eval threads and here is the blocklist" is a credibility win.

### Metrics

| # | Metric | How | Target |
|---|---|---|---|
| 1 | **Root-cause match** | Does system's top-3 include the confirmed fix? | top-3 accuracy |
| 2 | **Part correctness** | Does the named part match the one in the confirmed fix? | exact + fuzzy match rate |
| 3 | **Urgency safety** | On threads where the real answer was dangerous, did we say "do not drive"? | **false-safe rate = 0** |
| 4 | **Grounding** | Every price/phone/link exists in a real listing | invented-fact rate = 0 |
| 5 | **Listing dedup** | 100 hand-labelled listings, group them | vs keyword baseline |
| 6 | **VIN read** | 60 photos, clean and dirty | exact 17-char match; report 1-char-wrong separately |

### Baselines to beat

Report these next to your numbers or the eval means nothing:

- **Keyword search** on list.am — for metric 5
- **Bare LLM, no retrieval** — for metrics 1 and 2
- **Full system** — the thing you built

### Language split

Run metrics 1, 2 and 5 separately on **Armenian**, **Russian** and **English** inputs. Armenian is low-resource and models degrade on it. The gap is a real finding. Showing retrieval closing that gap is a stronger result than any single accuracy number.

### Judging summaries against forum answers

Don't only check the final text. Grade on:
- **Correct** — names the confirmed fix
- **Partially correct** — names it among other causes
- **Wrong** — misses it
- **Correctly abstained** — said "not enough information," when the thread was genuinely ambiguous

Abstention is a success, not a failure. Count it separately.

---

## 6. Safety

- Brakes, steering, suspension, airbags → always `do not drive`
- The app is not a mechanic. Say so in the UI.
- **Refuse rather than guess**: a low-confidence VIN read must say "retake the photo." A wrong VIN sends someone to buy a €400 part for the wrong car. Demo this refusal live.

---

## 7. Build order

| Hours | Do this |
|---|---|
| 0–1 | Confirm list.am and forums are scrapeable. **Go / no-go.** |
| 1–4 | Scrape forums. Split 70/30 immediately. Lock the eval set. |
| 4–8 | Case KB + embeddings. Agent 4 gate working end to end. |
| 8–14 | Agents 1–3. Listing dedup. VIN decode. |
| 14–18 | Run full eval. All baselines. Charts. |
| 18–21 | UI, image input, urgency rules |
| 21–23 | README, `.env.example`, seed script, rehearse demo |
| 23–24 | Buffer |

**Cache everything.** Do not scrape live during the demo. Venue wifi will fail.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Forums too small or no confirmed fixes | Check in hour 1. Fall back to mechanic-labelled cases. |
| Eval leakage | Hard blocklist, enforced in code |
| list.am blocks scraping | Cache early, rate-limit, keep a local snapshot |
| VIN decode needs a paid API | Find a free decoder in hour 1 or make VIN optional |
| Wrong safety call | Conservative urgency rules, hard-coded for safety-critical systems |
