import { useEffect, useRef, useState } from "react";
import "./App.css";

const STATUS_LABEL = {
  pending: "Pending",
  case_matched: "Case matched",
  researching: "Researching",
  complete: "Complete",
};

const EMPTY_FORM = { car_make: "", car_model: "", car_year: "", vin: "", raw_text: "" };

// The seven stages of the agentic graph, in the order they can run. `agent`
// marks the three that cost a model call -- everything else is deterministic
// and free, which is the point the pipeline view is meant to make visible.
// Labels are written for a car owner, not for whoever wrote the graph.
const PIPELINE = [
  { key: "vehicle", label: "Your car", detail: "Checking the VIN and which market it was built for" },
  { key: "route", label: "Your question", detail: "Working out what you need" },
  { key: "gate", label: "Past repairs", detail: "Looking for this problem already solved" },
  { key: "diagnose", label: "Diagnosis", detail: "Working out what's wrong" },
  { key: "parts", label: "Parts", detail: "Finding what to buy" },
  { key: "shops", label: "Mechanics", detail: "Finding who can fix it" },
  { key: "finalize", label: "Final checks", detail: "Making sure every claim has a source" },
];

const INTENT_LABEL = {
  diagnose: "find out what's wrong",
  part_lookup: "find a part to buy",
  shop_lookup: "find a mechanic",
  safety_check: "check if it's safe to drive",
};

// The trace notes are the graph's own engineering shorthand. These turn the
// ones a user might actually care about into plain sentences; anything not
// matched here stays in the collapsed technical section and never surfaces.
const NOTE_RULES = [
  [/^cache HIT \(top=([\d.]+) tier=(\w+)\)$/, (m) =>
    `Found a past repair for this problem (${Math.round(m[1] * 100)}% match on ${
      m[2] === "exact" ? "the same car" : m[2] === "near" ? "the same model" : "a similar car"
    })`],
  [/^cache miss \(no matches\)$/, () => "No past repair on file for this problem"],
  // A high score can still miss: a make-only ("loose") match may inform a
  // diagnosis but must never be served as one, so say which of the two reasons
  // it was rather than leaving "89% but not close enough" looking broken.
  [/^cache miss \(top=([\d.]+) tier=(\w+)\)$/, (m) => {
    const pct = Math.round(m[1] * 100);
    return m[2] === "loose"
      ? `Closest past repair was ${pct}% similar but on a different model — used as a hint, not as the answer`
      : `Nothing close enough in past repairs (closest was ${pct}%)`;
  }],
  [/^reused stored cases -- no model call$/, () =>
    "Answered from a confirmed past repair — no AI diagnosis needed"],
  [/^searched knowledge for '(.*?)'\s*->\s*(\d+)$/, (m) =>
    m[2] === "0"
      ? `Searched repair guides for "${m[1]}" — nothing found`
      : `Searched repair guides for "${m[1]}" — ${m[2]} found`],
  // The agent passes the same part in 5-6 spellings across three languages so
  // each finds different sellers. Listing them all is noise; the first plus a
  // count says the same thing.
  [/^searched listings for \[(.*)\]\s*->\s*(\d+)$/, (m) => {
    const terms = m[1].split(",").map((t) => t.trim().replace(/^'|'$/g, ""));
    const spellings = terms.length > 1 ? ` and ${terms.length - 1} other spellings` : "";
    return m[2] === "0"
      ? `Looked for "${terms[0]}"${spellings} — nothing for sale found`
      : `Looked for "${terms[0]}"${spellings} — ${m[2]} for sale`;
  }],
  [/^safety floor escalated urgency \((.*)\)$/, (m) =>
    `Safety rule: this involves the ${m[1]}, so it's marked do-not-drive`],
  [/^WARNING: (\w+) skipped the knowledge search$/, () =>
    "The assistant answered without checking repair guides"],
  [/^VIN rejected: (.*)$/, (m) => `VIN rejected — ${m[1]}`],
  [/^dropped (\d+) unresolvable reference/, (m) =>
    `Removed ${m[1]} claim(s) that couldn't be traced to a source`],
  [/^case store unavailable/, () => "Couldn't reach past repairs"],
  [/^(\w+) failed:/, (m) => `The ${m[1]} step failed`],
  [/^no model configured -- (.*)$/, (m) => `No AI configured — ${m[1]}`],
  [/^nothing to look up$/, () => "Nothing to look up"],
  [/^\d+ fitment warning/, (m) => m[0]],
  // Emitted before anything else when the question still needs translating.
  // Worth showing: it is the one step that can visibly get the question wrong.
  [/^understood as: (.+)$/, (m) => `Read your question as: "${m[1]}"`],
];

function humanizeNote(text) {
  const trimmed = text.trim();
  for (const [re, render] of NOTE_RULES) {
    const match = trimmed.match(re);
    if (match) return render(match);
  }
  return null; // engineering-only -- stays in the technical section
}

const URGENCY = [
  { label: "Drive it", cls: "u0" },
  { label: "Fix this week", cls: "u1" },
  { label: "Fix now", cls: "u2" },
  { label: "Do not drive", cls: "u3" },
];

const EMPTY_RUN = { events: [], answer: null, error: null, running: false };

// The trace notes are the graph's own words. Pulling the three structured
// facts out of them lets the UI show intent and the gate verdict while the run
// is still going, instead of waiting for the final answer.
function readTrace(events) {
  const notes = events.filter((e) => e.type === "note").map((e) => e.text);
  const find = (re) => notes.map((n) => n.trim().match(re)).find(Boolean);

  const gate = find(/^cache (HIT|miss) \((?:top=([\d.]+) tier=(\w+)|.*)\)$/);
  const intent = find(/^intent=(\S+)/);
  const reused = notes.some((n) => n.includes("reused stored cases"));
  const car = find(/^car=(.+?) region=(\S+) lang=(\S+) area=(\S+)$/);

  return {
    intent: intent && INTENT_LABEL[intent[1]],
    gate: gate && { hit: gate[1] === "HIT", score: gate[2], tier: gate[3] },
    reused,
    // "unknown vehicle" and region=?? are carmed's placeholders for "not told";
    // showing them as-is reads like an error, so they're simply omitted.
    car: car && car[1] !== "unknown vehicle" ? car[1] : null,
    story: notes.map(humanizeNote).filter(Boolean),
  };
}

function StatusDot({ state }) {
  return <span className={`stage-dot ${state}`} />;
}

// A step that has happened. The clock face reads as "this took time", which
// is the honest thing to say about a step that ran -- these are 2-8 second
// searches, not instant ticks.
function StepIcon() {
  return (
    <svg className="step-icon" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M8 5v3.2l2 1.2" fill="none" stroke="currentColor" strokeWidth="1.4"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Only ever the last row, and only once the run has actually finished.
function DoneIcon() {
  return (
    <svg className="step-icon" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M5.4 8.2l1.8 1.8 3.4-3.6" fill="none" stroke="currentColor" strokeWidth="1.5"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Rotates to point down when its section is open. Shared so the sidebar
// disclosure and the answer-provenance disclosure are visibly the same
// control, which "···" never was -- that reads as an overflow menu, and
// clicking it expecting rename/delete and getting a pipeline is a small lie.
function Chevron({ className = "" }) {
  return (
    <svg className={`chev ${className}`} viewBox="0 0 16 16" aria-hidden="true">
      <path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.75"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// What the collapsed row promises. Sources first, because they are the part
// worth opening for and the part the reader can check. A run that read
// nothing says "Steps" rather than "0 sources" -- a zero here is not a
// finding, it is just an empty section.
function traceLabel(events) {
  const n = sourcesFrom(events).length;
  return n ? `${n} source${n === 1 ? "" : "s"}` : "Steps";
}

// Sources the run actually opened, newest wording last, de-duplicated. Read
// off the persisted trace, so a request from last week still shows what it
// was working from -- not just the one currently streaming.
function sourcesFrom(events) {
  return Object.values(
    events
      .filter((e) => e.type === "partial" && e.kind === "source")
      .reduce((acc, e) => ({ ...acc, [e.data.id]: e.data }), {})
  );
}

// Rendered only while its history record is expanded, so everything it knows is
// laid out at once -- the "..." on the record is the only toggle.
function Pipeline({ run }) {
  const { events, running } = run;
  const started = events.filter((e) => e.type === "step").map((e) => e.name);
  const { intent, car, reused } = readTrace(events);
  const sources = sourcesFrom(events);
  const current = started[started.length - 1];
  // Not "has a done event": a trace replayed from the server carries the steps
  // but not the terminal event, and treating that as unfinished would leave
  // every stage the graph skipped showing as still-pending.
  const finished = !running && events.length > 0;

  const stateOf = (key) => {
    if (!started.includes(key)) return finished ? "skipped" : "idle";
    return key === current && running ? "running" : "done";
  };

  // What a step is actually doing, which is not always its static blurb.
  const detailOf = (stage) => {
    if (stage.key === "route" && intent) return `You want to ${intent}`;
    // On a gate hit this step still runs, but it copies the stored fix instead
    // of asking the model. Saying "working out what's wrong" would misrepresent it.
    if (stage.key === "diagnose" && reused) {
      return "Recalled from the past repair — the AI wasn't asked";
    }
    return stage.detail;
  };

  return (
    <div className="pipeline">
      <div className="pipeline-head">
        <span className="pipeline-title">
          {running ? "Running…" : finished ? "Pipeline complete" : "Pipeline"}
        </span>
      </div>

      {car && <div className="pipeline-sub">{car}</div>}

      {/* One row per stage, carrying its own detail. This used to be two full
          passes over PIPELINE -- chips, then the same seven labels again with
          descriptions under them -- which read as fourteen things happening
          instead of seven. */}
      <div className="stages">
        {PIPELINE.map((stage) => {
          const state = stateOf(stage.key);
          return (
            <div key={stage.key} className={`stage ${state}`}>
              <StatusDot state={state} />
              <div className="stage-body">
                <span className="stage-label">{stage.label}</span>
                {/* Only for the stages that ran. A description of a step that
                    was skipped is filler. */}
                {state !== "skipped" && state !== "idle" && (
                  <span className="stage-detail">{detailOf(stage)}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="pipeline-details">
        {/* No gate card here. "New problem for us / Nothing matching in our
            past repairs" was a bordered, coloured panel restating one line
            the run narrative already carries -- decoration around a fact,
            competing for attention with the sources, which are the part
            worth reading. The verdict still shows in the answer panel. */}
        {/* The run's narrative lives in the answer panel now, where the user is
            actually looking. Repeating it here was the same text twice. The
            sources do belong here: a count of what was read is not checkable,
            and the whole point of showing provenance is that you can go look. */}
        {sources.length > 0 && (
          <div className="src-list">
            <div className="src-head">Sources read <span>{sources.length}</span></div>
            {sources.map((s) => (
              <a
                key={s.id}
                className="src"
                href={s.url}
                target="_blank"
                rel="noreferrer noopener"
                title={s.url}
              >
                <span className="src-kind">{s.kind}</span>
                <span className="src-title">{s.title}</span>
              </a>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}

// `latest` is the last answer in the conversation. Only it carries the urgency
// verdict: "do not drive" is advice about the car as it stands now, and three
// of them stacked down the thread -- two of them superseded -- is three
// conflicting instructions rather than a history.
// "understand" is emitted by the view before the graph starts, so it has no
// pipeline stage of its own.
const STAGE_DETAIL = Object.fromEntries([
  ["understand", "Reading your question"],
  ["reply", "Putting the answer into your language"],
  ...PIPELINE.map((s) => [s.key, s.detail]),
]);

// What the user watches while the run is going.
//
// Two things stream in. The graph's own progress notes, which read as the
// steps it is taking; and finished pieces of the answer -- the diagnostician
// writes one cause at a time, and each appears the moment its JSON object
// closes rather than after the whole reply lands.
//
// Nothing streamed here is authoritative. A partial that never completes, or
// one from an attempt the model then abandoned, must not be able to change
// what the run produced -- so once the run finishes, the streamed causes and
// parts give way to AnswerCard's verified copies.
//
// The record of *how* the answer was reached does not give way: the steps and
// the sources read stay on screen afterwards. Watching six sources go by and
// then having them vanish the moment the answer lands is worse than never
// showing them, because the answer then looks like it came from nowhere.
function LiveAnswer({ run }) {
  const { events, running } = run;
  const steps = events.filter((e) => e.type === "step").map((e) => e.name);
  const detail = running
    ? STAGE_DETAIL[steps[steps.length - 1]] || "Starting…"
    : "Research";
  const { story } = readTrace(events);
  // The model can write a full set of causes, call a tool, and then come back
  // with something else entirely -- a clarifying question instead of an
  // answer. A "reset" marks that abandonment, and everything of that kind
  // before it has to disappear, or the user is left reading a diagnosis the
  // run then withdrew.
  const since = (kind) => {
    const all = events.filter(
      (e) => e.type === "partial" && (e.kind === kind || (e.kind === "reset" && e.data?.of === kind))
    );
    const last = all.map((e) => e.kind).lastIndexOf("reset");
    return all.slice(last + 1).map((e) => e.data);
  };
  const causes = since("cause");
  const parts = since("part");
  const sources = sourcesFrom(events);
  // Retrieved is not the same as used. Only what a cause actually points at
  // is a citation, and finalize can still drop one -- so this is a preview.
  const cited = new Set(causes.flatMap((c) => c.evidence || []));

  // Open while the run is going -- that is the whole point of it -- and folds
  // itself away once the answer lands, so the transcript stays readable
  // without the record being thrown away. Reopening is one click, and the
  // choice sticks until the next run starts.
  const [open, setOpen] = useState(true);
  useEffect(() => setOpen(running), [running]);

  // What the collapsed row has to say for itself. "6 steps" alone is not worth
  // opening; the number of sources is.
  const summary = [
    sources.length && `${sources.length} source${sources.length === 1 ? "" : "s"}`,
    story.length && `${story.length} step${story.length === 1 ? "" : "s"}`,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className={`live ${running ? "" : "settled"} ${open ? "open" : "closed"}`}>
      <button
        type="button"
        className="live-head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {/* Always present, running or not: it is what tells you the row can be
            folded, and a control that appears only after the fact reads as a
            different control. */}
        <Chevron className="live-chevron" />
        {running && (
          <span className="pending-dots">
            <i />
            <i />
            <i />
          </span>
        )}
        <span className="live-title">{detail}</span>
        {summary && <span className="live-summary">{summary}</span>}
      </button>

      {/* One wrapper for the whole body, so open/close is a single height
          transition rather than four sections popping independently. */}
      <div className={`collapse ${open ? "open" : ""}`}>
       <div className="collapse-inner">
      {story.length > 0 && (
        <ol className="live-story">
          {story.map((line, i) => (
            <li key={i} className={i === story.length - 1 && running ? "fresh" : ""}>
              <StepIcon />
              <span className="step-text">{line}</span>
            </li>
          ))}
          {/* Only once the run has really ended. A "Done" that appears while
              work is still going is the one thing this list must not say. */}
          {!running && (
            <li className="step-done">
              <DoneIcon />
              <span className="step-text">Done</span>
            </li>
          )}
        </ol>
      )}

      {sources.length > 0 && (
        <div className="live-block">
          <div className="answer-section">
            Sources read <span className="live-count">{sources.length}</span>
          </div>
          <div className="live-sources">
            {sources.map((s) => (
              <a
                key={s.id}
                className={`live-source appearing ${cited.has(`doc:${s.id}`) ? "cited" : ""}`}
                href={s.url}
                target="_blank"
                rel="noreferrer noopener"
                title={s.url}
              >
                <span className="live-source-kind">{s.kind}</span>
                {s.title}
              </a>
            ))}
          </div>
        </div>
      )}

      {running && causes.length > 0 && (
        <div className="live-block">
          <div className="answer-section">Likely causes</div>
          {causes.map((c, i) => (
            <div key={i} className="cause appearing">
              <div className="cause-title">
                {i + 1}. {c.title}
                {c.confidence && <span className={`conf ${c.confidence}`}>{c.confidence}</span>}
              </div>
              {c.explanation && <div className="cause-why">{c.explanation}</div>}
              {c.evidence?.length > 0 && (
                <div className="evidence">
                  <span className="evidence-label">Based on</span>
                  {c.evidence.map((e) => {
                    const [kind, id] = e.split(":");
                    return (
                      <span key={e} className="ref">
                        {kind === "case" ? `past repair #${id}` : `source ${id}`}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {running && parts.length > 0 && (
        <div className="live-block">
          <div className="answer-section">What to buy</div>
          <div className="live-parts">
            {/* Names only. Prices and links are resolved from the stored
                listing records in the final answer, never from a partial. */}
            {parts.map((p, i) => (
              <span key={i} className="live-part appearing">
                {p.name_en}
              </span>
            ))}
          </div>
        </div>
      )}
       </div>
      </div>
    </div>
  );
}

function AnswerCard({ answer, latest = true, at }) {
  if (!answer) return null;
  const urgency = URGENCY[answer.urgency] || URGENCY[1];

  return (
    <div className={`answer-card ${latest ? "" : "earlier"}`}>
      {latest ? (
        <div className={`urgency-banner ${urgency.cls}`}>
          <strong>{urgency.label}</strong>
          {answer.urgency_reason && <span>{answer.urgency_reason}</span>}
        </div>
      ) : (
        <div className="answer-when">
          Earlier answer
          {at &&
            ` · ${new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`}
        </div>
      )}

      {/* answer.message is deliberately not rendered here -- it is the
          assistant's chat bubble just above this card. */}
      {answer.causes.length > 0 && (
        <>
          <div className="answer-section">Likely causes</div>
          {answer.causes.map((c, i) => (
            <div key={i} className="cause">
              <div className="cause-title">
                {i + 1}. {c.title}
                <span className={`conf ${c.confidence}`}>{c.confidence}</span>
              </div>
              {c.explanation && <div className="cause-why">{c.explanation}</div>}
              {c.evidence.length > 0 ? (
                <div className="evidence">
                  <span className="evidence-label">Based on</span>
                  {c.evidence.map((e) => {
                    const [kind, id] = e.split(":");
                    return (
                      <span key={e} className="ref">
                        {kind === "case" ? `past repair #${id}` : `source ${id}`}
                      </span>
                    );
                  })}
                </div>
              ) : (
                // Must never be mistaken for a sourced cause. The assistant is
                // allowed to give the textbook differential when the sources
                // don't cover this car -- but it has to say that's what it is.
                <div className="evidence unsourced">
                  <span className="evidence-label">General knowledge</span>
                  <span className="ref muted">no source found for this car</span>
                </div>
              )}
            </div>
          ))}
        </>
      )}

      {answer.repair_steps.length > 0 && (
        <>
          <div className="answer-section">Repair steps</div>
          <ul className="steps">
            {answer.repair_steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </>
      )}

      {/* What to buy. Each PartOption's listing_ids resolve against
          `answer.listings`, so prices and links come from the record rather
          than from anything the model wrote. A listing whose page was never
          opened has no price -- shown as "ask seller" rather than left blank,
          because an empty price reads as free. */}
      {answer.parts?.options?.length > 0 && (
        <>
          <div className="answer-section">What to buy</div>
          {answer.parts.options.map((opt, i) => (
            <div className="part" key={i}>
              <div className="part-name">
                {opt.name_en}
                {opt.part_numbers?.length > 0 && (
                  <span className="part-nums"> &middot; {opt.part_numbers.join(", ")}</span>
                )}
              </div>
              {opt.aliases?.length > 0 && (
                <div className="part-aliases">also sold as: {opt.aliases.join(" \u00b7 ")}</div>
              )}
              <ul className="listings">
                {(opt.listing_ids || []).map((lid) => {
                  const l = answer.listings?.[lid];
                  if (!l) return null;
                  return (
                    <li key={lid}>
                      <a href={l.url} target="_blank" rel="noreferrer">{l.title}</a>
                      <span className="price">
                        {l.price
                          ? Math.round(l.price.amount).toLocaleString() + " " + l.price.currency
                          : "ask seller"}
                      </span>
                      {l.condition && <span className="cond">{l.condition}</span>}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </>
      )}

      {/* Fitment warnings sit with the parts deliberately: they qualify a
          specific listing, and separating them from what they qualify is how
          someone buys the wrong part. */}
      {answer.parts?.fitment_warnings?.length > 0 && (
        <div className="fitment">
          {answer.parts.fitment_warnings.map((w, i) => (
            <div key={i} className={w.severity === "blocking" ? "fit-block" : "fit-check"}>
              {w.severity === "blocking" ? "Does not fit" : "Check fitment"} &mdash; {w.message}
            </div>
          ))}
        </div>
      )}

      {/* Who to call. The phone number is the most actionable thing on the
          page, and every shop has one -- the research layer drops any record
          whose number it cannot read. */}
      {answer.shops?.length > 0 && (
        <>
          <div className="answer-section">Who to call</div>
          <ul className="shops">
            {answer.shops.map((s) => (
              <li key={s.id}>
                <span className="shop-name">{s.name}</span>
                <a className="shop-phone" href={"tel:" + s.phone}>{s.phone}</a>
                {s.area && <span className="shop-area">{s.area}</span>}
              </li>
            ))}
          </ul>
        </>
      )}

      <div className="answer-foot">
        {answer.from_cache && <span className="ok">From a confirmed past repair</span>}
        <span className={answer.dropped_refs === 0 ? "ok" : "bad"}>
          {answer.dropped_refs === 0
            ? "Every claim traced to a source"
            : `${answer.dropped_refs} unsupported claim(s) removed`}
        </span>
        <span className="disclaimer">{answer.disclaimer}</span>
      </div>
    </div>
  );
}

// Exactly one response per run, and prose wins.
//
// When the graph sets `message` it is telling the user something other than a
// diagnosis -- a clarifying question, an abstention, a rejected VIN -- and that
// text is already in the transcript as an assistant message. A card alongside
// it would be a second answer to the same question. Checking only for causes
// and steps was not enough: a needs_clarification run still returns
// repair_steps, so five of them were rendering a bubble *and* a card.
// Does this run have anything structured to show, regardless of what it also
// said in prose?
//
// This used to be "has no message", on the assumption that a message meant a
// clarifying question and a clarifying question meant nothing else was found.
// Both halves are now false: a run can name three cited causes, price the
// parts, list a workshop, and still ask one question to narrow it further --
// and the old test threw all of that away at render time because the question
// existed.
function hasCard(answer) {
  if (!answer) return false;
  return Boolean(
    answer.causes?.length ||
      answer.repair_steps?.length ||
      answer.parts?.options?.length ||
      answer.shops?.length
  );
}

// One chronological list of what the user asked and what each run answered.
// The two live in separate collections server-side, so they are zipped back
// together by timestamp rather than rendered as two stacked blocks.
function buildTimeline(active) {
  const answers = active.runs || [];
  const lastId = answers.length ? answers[answers.length - 1].id : null;

  const entries = [
    // Both roles. The assistant's turns are the answers' prose, saved as
    // messages by the run endpoint, so a clarifying question stays in the
    // transcript instead of living only on the answer it came with.
    ...active.messages.map((m) => ({
      key: `m-${m.id}`,
      at: m.created_at,
      rank: 0, // ahead of the run row written in the same instant
      node: (
        <div className={`message-row ${m.role}`}>
          <div className={`bubble ${m.role}`}>{m.content}</div>
        </div>
      ),
    })),
    // The structured half: causes, repair steps, and the urgency verdict. Its
    // prose is already the bubble above, so a run with nothing structured to
    // add -- a clarifying question -- contributes no card at all.
    ...answers
      .filter((r) => hasCard(r.answer))
      .map((r) => ({
        key: `a-${r.id}`,
        at: r.created_at,
        rank: 1,
        node: <AnswerCard answer={r.answer} latest={r.id === lastId} at={r.created_at} />,
      })),
  ];

  // Requests answered before runs were kept have a summary but no run rows.
  // Without this their one answer would vanish from the thread entirely.
  if (answers.length === 0 && active.summary && hasCard(active.summary)) {
    entries.push({
      key: "a-legacy",
      at: active.updated_at,
      rank: 1,
      node: <AnswerCard answer={active.summary} />,
    });
  }

  return entries.sort((a, b) => new Date(a.at) - new Date(b.at) || a.rank - b.rank);
}

// A destructive confirm that belongs to the app rather than the browser.
// Escape and a backdrop click both cancel, and the cancel button takes focus
// on open -- so the reflex "hit enter to get rid of this" is the safe answer,
// not the irreversible one.
function ConfirmDialog({ title, body, confirmLabel, onConfirm, onCancel }) {
  const cancelRef = useRef(null);

  useEffect(() => {
    cancelRef.current?.focus();
    const onKey = (e) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div
        className="modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-title" id="confirm-title">
          {title}
        </div>
        <div className="modal-body">{body}</div>
        <div className="modal-actions">
          <button type="button" className="modal-btn" ref={cancelRef} onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="modal-btn danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.status === 204 ? null : res.json();
}

function uid() {
  return `local-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function LogoMark({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <rect width="24" height="24" rx="7" fill="var(--accent)" />
      <path
        d="M3 12h3.5l1.4-3 2 6 1.4-3H21"
        stroke="var(--accent-fg)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M4 12L20 4L13 20L11 13L4 12Z" fill="currentColor" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  );
}

export default function App() {
  const [requests, setRequests] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [active, setActive] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [creating, setCreating] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [sending, setSending] = useState(false);
  // Keyed by request id rather than a single slot: switching records used to
  // wipe the run, so an open pipeline vanished and could only be brought back
  // by re-running the diagnosis. Each record now keeps its own.
  const [runs, setRuns] = useState({});
  // Which records have their steps expanded. Never cleared -- an open panel
  // stays open for the rest of the session, however much you click around.
  const [openSteps, setOpenSteps] = useState(() => new Set());
  const messagesRef = useRef(null);

  const activeRun = runs[activeId] || EMPTY_RUN;

  const loadRequests = () => api("/api/diagnostics/").then((data) => setRequests(data.results));

  // Seeds the record's run from the trace the server persisted, so a pipeline
  // survives a reload and shows up on old records too. A run still streaming is
  // left alone -- the live events are ahead of what's been saved.
  const loadActive = (id) =>
    api(`/api/diagnostics/${id}/`).then((data) => {
      setActive(data);
      setRuns((prev) =>
        prev[id]?.running
          ? prev
          : { ...prev, [id]: { ...EMPTY_RUN, events: data.trace || [], answer: data.summary } }
      );
      return data;
    });

  // Confirmed before it happens: it takes the conversation and every run with
  // it, and there is nowhere to get them back from. Holds the whole record
  // rather than the id, so the dialog can name the car being deleted.
  const [pendingDelete, setPendingDelete] = useState(null);

  const deleteRequest = async () => {
    const id = pendingDelete?.id;
    setPendingDelete(null);
    if (!id) return;
    await api(`/api/diagnostics/${id}/`, { method: "DELETE" });
    setRuns((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    if (id === activeId) setActiveId(null);
    await loadRequests();
  };

  const toggleSteps = (id) =>
    setOpenSteps((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  useEffect(() => {
    loadRequests().catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    if (activeId) loadActive(activeId).catch((err) => console.error(err));
    else setActive(null);
  }, [activeId]);

  useEffect(() => {
    // Follows the pipeline too, not just new messages -- it now grows below the
    // transcript as each stage reports in.
    if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [active?.messages, active?.runs, activeRun.events.length, activeRun.running]);

  // Consumes the SSE feed from the run endpoint, appending each pipeline event
  // as it arrives so the stages light up while the graph is still working.
  const runPipeline = async (id) => {
    const patch = (fn) => setRuns((prev) => ({ ...prev, [id]: fn(prev[id] || EMPTY_RUN) }));
    setRuns((prev) => ({ ...prev, [id]: { ...EMPTY_RUN, running: true } }));
    try {
      const res = await fetch(`/api/diagnostics/${id}/run/stream/`, { method: "POST" });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Request failed (${res.status})`);
      }
      if (!res.body) throw new Error("The server sent no data.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const line = buffer.slice(0, idx).replace(/^data:\s*/, "");
          buffer = buffer.slice(idx + 2);
          if (!line) continue;
          const event = JSON.parse(line);

          if (event.type === "error") {
            patch((prev) => ({ ...prev, running: false, error: event.message }));
            return;
          }
          patch((prev) => ({
            ...prev,
            events: [...prev.events, event],
            answer: event.type === "done" ? event.answer : prev.answer,
          }));
        }
      }
      patch((prev) => ({ ...prev, running: false }));
      await Promise.all([loadActive(id), loadRequests()]);
    } catch (err) {
      patch((prev) => ({ ...prev, running: false, error: err.message }));
    }
  };

  const createRequest = async (e) => {
    e.preventDefault();
    setCreating(true);
    try {
      const body = {
        car_make: form.car_make,
        car_model: form.car_model,
        car_year: form.car_year ? Number(form.car_year) : null,
        // Optional: when present, vPIC decodes factory engine/trim so parts
        // and NHTSA lookups target the exact build instead of a guess.
        vin: form.vin.trim().toUpperCase() || "",
        raw_text: form.raw_text,
      };
      const req = await api("/api/diagnostics/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setForm(EMPTY_FORM);
      await loadRequests();
      setActiveId(req.id);
      // The question has been asked -- show the pipeline answering it.
      runPipeline(req.id);
    } finally {
      setCreating(false);
    }
  };

  const sendMessage = async () => {
    const content = chatInput.trim();
    if (!content || !activeId || sending) return;
    setChatInput("");
    setSending(true);

    // The turn shows straight away; the answer to it is the pipeline's, so
    // nothing is appended on the assistant's side here.
    // created_at matters: the timeline is sorted by it, and an entry without
    // one sorts as an invalid date and lands in the wrong place.
    setActive((prev) => ({
      ...prev,
      messages: [
        ...prev.messages,
        { id: uid(), role: "user", content, created_at: new Date().toISOString() },
      ],
    }));

    try {
      // Persists the turn and re-summarizes the conversation, so the run below
      // is decided against what the user has just clarified.
      await api(`/api/diagnostics/${activeId}/turns/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
    } catch (err) {
      setRuns((prev) => ({
        ...prev,
        [activeId]: { ...(prev[activeId] || EMPTY_RUN), error: err.message },
      }));
      setSending(false);
      return;
    }

    // `sending` stays true until the run has taken over, so the placeholder
    // can't blink back to the stale answer in the handover between the two.
    try {
      await runPipeline(activeId);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <LogoMark />
          <span className="wordmark">
            Car<span className="accent-text">Med</span>
          </span>
        </div>

        <button className="new-chat-btn" onClick={() => setActiveId(null)}>
          <PlusIcon /> New diagnosis
        </button>

        <div className="req-list">
          <div className="req-list-label">History</div>
          {requests.length === 0 && <small className="hint">No requests yet.</small>}
          {requests.map((r) => {
            // Live run if there is one, otherwise the trace the server saved for
            // this record -- so every record that has ever been diagnosed keeps
            // its "...", not just the one being viewed.
            const rowRun =
              runs[r.id] ||
              (r.trace?.length ? { ...EMPTY_RUN, events: r.trace, answer: r.summary } : null);
            const hasRun = rowRun && (rowRun.events.length > 0 || rowRun.running);
            const open = hasRun && openSteps.has(r.id);
            return (
              <div key={r.id} className="req-entry">
                <div
                  className={`req-item ${r.id === activeId ? "active" : ""}`}
                  onClick={() => setActiveId(r.id)}
                >
                  <span className={`dot ${r.status}`} />
                  <span className="title">
                    {r.car_make} {r.car_model} {r.car_year || ""}
                  </span>
                  {/* Labelled with what it opens, rather than an arrow.
                      A chevron in a list of records reads as "go into this",
                      which is what clicking the row already does, and it
                      leaves every row looking identical at rest. The count is
                      the useful part: it says which of these answers actually
                      had something behind it before you open any of them. */}
                  {hasRun && (
                    <button
                      type="button"
                      className={`req-trace ${open ? "open" : ""}`}
                      aria-expanded={open}
                      title={open ? "Hide the research behind this answer" : "The research behind this answer"}
                      onClick={(e) => {
                        e.stopPropagation(); // the row itself only selects
                        toggleSteps(r.id);
                      }}
                    >
                      {open ? "Hide" : traceLabel(rowRun.events)}
                    </button>
                  )}
                  <button
                    type="button"
                    className="req-del"
                    title="Delete this diagnosis"
                    aria-label="Delete this diagnosis"
                    onClick={(e) => {
                      e.stopPropagation();
                      setPendingDelete(r);
                    }}
                  >
                    ×
                  </button>
                </div>
                {/* Rendered whether open or not: a height transition needs
                    something to measure, and mounting on open would snap. */}
                {hasRun && (
                  <div className={`collapse ${open ? "open" : ""}`}>
                    <div className="collapse-inner">
                      <Pipeline run={rowRun} />
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </aside>

      <main className="main">
        {!active && (
          <div className="welcome">
            <div className="welcome-card">
              <LogoMark size={40} />
              <h1>
                Car<span className="accent-text">Med</span>
              </h1>
              <p className="tagline">Describe what's wrong with your car and get instant help.</p>

              <form className="welcome-form" onSubmit={createRequest}>
                <div className="row">
                  <div>
                    <label>Make</label>
                    <input
                      value={form.car_make}
                      onChange={(e) => setForm({ ...form, car_make: e.target.value })}
                      placeholder="Toyota"
                    />
                  </div>
                  <div>
                    <label>Model</label>
                    <input
                      value={form.car_model}
                      onChange={(e) => setForm({ ...form, car_model: e.target.value })}
                      placeholder="Camry"
                    />
                  </div>
                  <div>
                    <label>Year</label>
                    <input
                      type="number"
                      value={form.car_year}
                      onChange={(e) => setForm({ ...form, car_year: e.target.value })}
                      placeholder="2015"
                    />
                  </div>
                </div>
                <label>
                  VIN <span style={{ fontWeight: 400 }}>(optional)</span>
                </label>
                <input
                  value={form.vin}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      vin: e.target.value.toUpperCase().replace(/[^A-HJ-NPR-Z0-9]/gi, "").slice(0, 17),
                    })
                  }
                  placeholder="1N4AL3AP7FC123456"
                  maxLength={17}
                  autoComplete="off"
                  spellCheck={false}
                  title="Improves diagnosis: exact engine and trim from the factory catalogue"
                />
                <label>What's going on?</label>
                {/* Enter submits, shift+enter breaks the line -- the same deal
                    the chat composer makes. The other fields already submit on
                    enter because this is a real form; a textarea is the one
                    control where the browser keeps the newline, and it is the
                    field people are actually typing in. */}
                <textarea
                  required
                  value={form.raw_text}
                  onChange={(e) => setForm({ ...form, raw_text: e.target.value })}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey && !creating && form.raw_text.trim()) {
                      e.preventDefault();
                      e.currentTarget.form?.requestSubmit();
                    }
                  }}
                  placeholder="Grinding noise when braking at low speed..."
                />
                <button
                  className="primary-btn"
                  type="submit"
                  disabled={creating || !form.raw_text.trim()}
                >
                  {creating ? "Starting…" : "Start diagnosis"}
                </button>
              </form>
            </div>
          </div>
        )}

        {active && (
          <>
            <div className="header-bar">
              <LogoMark size={22} />
              <span className="vehicle">
                {active.car_make} {active.car_model} {active.car_year || ""}
                {active.vin ? (
                  <span
                    style={{
                      marginLeft: 8,
                      opacity: 0.7,
                      fontSize: "0.85em",
                      fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                    }}
                  >
                    · {active.vin}
                  </span>
                ) : null}
              </span>
              <span className={`pill ${active.status}`}>{STATUS_LABEL[active.status] || active.status}</span>

              <button
                className="primary-btn compact"
                onClick={() => runPipeline(activeId)}
                disabled={activeRun.running}
                title="Re-run the agentic pipeline over the conversation so far"
              >
                {activeRun.running ? "Running…" : "Run diagnosis"}
              </button>
            </div>

            {active.case_matches.length > 0 && (
              <div className="match-card">
                <div className="match-title">
                  Past diagnoses of this car the assistant is working from
                </div>
                {active.case_matches.map((m) => (
                  <div key={m.id} className="match-row">
                    <span className="confidence">{Math.round(m.confidence * 100)}%</span> —{" "}
                    {m.matched.symptom_text}
                    {/* A past case with no recorded outcome is still worth
                        showing -- it is the same car with the same symptom --
                        but it must not render as an arrow pointing at nothing. */}
                    {m.matched.fix && <> → {m.matched.fix}</>}
                  </div>
                ))}
              </div>
            )}

            <div className="messages" ref={messagesRef}>
              <div className="messages-inner">
                {/* Questions and the answers they produced, in the order they
                    happened. Assistant chat turns are filtered out: records
                    made before that was removed still carry them, and they
                    would sit alongside the real answer saying something else. */}
                {buildTimeline(active).map((entry) => (
                  <div key={entry.key}>{entry.node}</div>
                ))}

                {activeRun.error && <div className="run-error">{activeRun.error}</div>}

                {/* Shows the run as it happens, then stays as the record of
                    where the answer came from. It must not be conditioned on
                    `running` alone: that unmounts the whole thing the instant
                    the answer lands, taking the sources with it. */}
                {(sending || activeRun.running || activeRun.events.length > 0) && (
                  <LiveAnswer run={activeRun} />
                )}
              </div>
            </div>

            <div className="composer-wrap">
              <div className="composer">
                <textarea
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder={
                    activeRun.running ? "Re-running the diagnosis…" : "Answer or ask a follow-up…"
                  }
                  disabled={sending || activeRun.running}
                  rows={1}
                />
                <button
                  className="send-btn"
                  onClick={sendMessage}
                  disabled={sending || activeRun.running}
                >
                  <SendIcon />
                </button>
              </div>
            </div>
          </>
        )}
      </main>

      {pendingDelete && (
        <ConfirmDialog
          title="Delete this diagnosis?"
          body={
            <>
              <strong>
                {pendingDelete.car_make} {pendingDelete.car_model}{" "}
                {pendingDelete.car_year || ""}
              </strong>{" "}
              and everything in it — the conversation, the answers, and the steps behind
              them. This cannot be undone.
            </>
          }
          confirmLabel="Delete"
          onConfirm={() => deleteRequest().catch((err) => console.error(err))}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
  );
}
