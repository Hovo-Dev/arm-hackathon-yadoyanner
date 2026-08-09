# CarMed

**Car diagnosis for Armenia** — ARM LLM Hackathon 2026

Describe your car problem however you type it. Get a diagnosis with sources, parts at real prices, and a mechanic to call.

![CarMed](docs/carmed-ui.png)

Pitch deck: [`docs/carmed-deck.html`](docs/carmed-deck.html)

---

## The problem

In Armenia, a strange noise usually means guessing, posting on a forum in a language you may not write well, or handing the car to a mechanic and accepting whatever you are told.

The fleet is mostly older US and European imports. Good repair knowledge lives in foreign forums and manuals. Parts and workshops live on **list.am**, **auto.am**, and **turn.am**. Nothing connects those worlds.

And the question often arrives like this:

```text
Mexanika karobki pervi vaxt dzena galis
```

Armenian typed on a Latin keyboard, mixed with Russian loanwords — not an edge case; it’s how people write.

## What CarMed does

One sentence in → four answers out:

1. **What’s probably wrong** — up to three causes, ranked, each pointing at a source  
2. **How urgent** — from “drive it” to “do not drive” (brakes and steering can only escalate, never soften)  
3. **What to buy** — real listings from list.am and auto.am, with the seller’s own price  
4. **Who to call** — a workshop from turn.am (name, address, phone)

Optional VIN improves identity via the official US vehicle catalogue (exact engine and trim), so later steps target the right build.

## Where it looks

Not “just ask an LLM” and not open-web chaos:

| Layer | What |
|---|---|
| Official | NHTSA owner complaints and recalls for that year/make/model — symptom paired with what dealers actually found |
| Identity | vPIC VIN decode when the owner provides a VIN |
| Web | Forums and guides, restricted to an allowlist |
| Local market | Armenian parts and workshops |
| Memory | Past completed diagnoses on similar cars (case matching) |

## Why it can be trusted

A better model alone does not fix these. CarMed is built so answers can be *checked*:

- **It cannot invent a price.** Parts results carry listing ids; every price shown was read from a seller’s listing.  
- **Claims need sources.** Unsupported citations are removed before the user sees them.  
- **The run is visible.** Searches and sources appear as they happen; you can open the links the diagnosis actually used.

## Evaluation

Three real faults with the repair confirmed, compared to Claude with web search:

| Car & symptom | What it actually was | Claude | CarMed |
|---|---|---|---|
| Altima 2014 — SRS light after an oil change | B0020 — seat back replaced | partial | **hit** (1st, with code and part) |
| Altima 2014 — stiff steering, no warning light | Both front control arms | missed | **hit** (citing matching complaint) |
| Civic 2016 — sticky steering after 15 min | Electric power steering rack | hit | **hit** (1st) |

**CarMed 3 / 3 · Claude with web search 1.5 / 3**

Every number in the deck is measured from a real run.

## What makes this worth building

Models keep getting better at cars. That does not give you:

- **Live local prices** — what a control arm costs on list.am *this week*  
- **A phone number** — the workshop in Yerevan that does this job  
- **Enforced provenance** — dropping claims nothing supports  
- **Repairs confirmed here** — finished diagnoses become cases for the next owner

## Honest limits

- US regulator data does not cover every European car on Armenian roads  
- Source ranking is still rough  
- Latin-script Armenian coverage has gaps  

## What’s next

- Stronger automatic evaluation (LLM-as-judge on a held-out set)  
- Phone-first product  
- Fault data beyond US recalls  
- Photo input (VIN plate, warning lights, worn parts)

## Disclaimer

CarMed is not a substitute for a professional mechanic. Confirm before buying parts or driving a car you believe is unsafe.
