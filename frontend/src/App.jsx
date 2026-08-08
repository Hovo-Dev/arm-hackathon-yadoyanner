import { useEffect, useRef, useState } from "react";
import "./App.css";

const IMAGE_TYPES = [
  ["vin_plate", "VIN plate"],
  ["part", "Old part"],
  ["dashboard_light", "Dashboard light"],
  ["damage", "Damage / leak"],
];

const STATUS_LABEL = {
  pending: "Pending",
  case_matched: "Case matched",
  researching: "Researching",
  complete: "Complete",
};

const EMPTY_FORM = { car_make: "", car_model: "", car_year: "", raw_text: "" };

// The seven stages of the agentic graph, in the order they can run. `agent`
// marks the three that cost a model call -- everything else is deterministic
// and free, which is the point the pipeline view is meant to make visible.
// Labels are written for a car owner, not for whoever wrote the graph.
const PIPELINE = [
  { key: "vehicle", label: "Your car", detail: "Checking the VIN and which market it was built for" },
  { key: "route", label: "Your question", detail: "Working out what you need", agent: "router" },
  { key: "gate", label: "Past repairs", detail: "Looking for this problem already solved" },
  { key: "diagnose", label: "Diagnosis", detail: "Working out what's wrong", agent: "diagnostician" },
  { key: "parts", label: "Parts", detail: "Finding what to buy", agent: "parts_explorer" },
  { key: "shops", label: "Mechanics", detail: "Finding who can fix it" },
  { key: "finalize", label: "Final checks", detail: "Making sure every claim has a source" },
];

const INTENT_LABEL = {
  diagnose: "find out what's wrong",
  part_lookup: "find a part to buy",
  shop_lookup: "find a mechanic",
  safety_check: "check if it's safe to drive",
};

// What each data source is, in the user's terms rather than the method name.
const SOURCE_LABEL = {
  "case_store.find_similar": "Past repairs we've seen",
  "research.search_knowledge": "Repair guides and forums",
  "research.search_listings": "Parts for sale",
  "research.search_shops": "Repair shops",
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
    notes,
  };
}

function StatusDot({ state }) {
  return <span className={`stage-dot ${state}`} />;
}

// Rendered only while its history record is expanded, so everything it knows is
// laid out at once -- the "..." on the record is the only toggle.
function Pipeline({ run }) {
  const { events, running } = run;
  const started = events.filter((e) => e.type === "step").map((e) => e.name);
  const llm = events.filter((e) => e.type === "llm").map((e) => e.agent);
  // One row per source, not per call: the parts agent hits the same source five
  // or six times with different wording, and six identical "nothing found" rows
  // say no more than one does.
  const lookups = Object.values(
    events
      .filter((e) => e.type === "lookup")
      .reduce((acc, e) => {
        const row = acc[e.name] || (acc[e.name] = { name: e.name, count: 0, calls: 0 });
        row.count += e.count;
        row.calls += 1;
        return acc;
      }, {})
  );
  const { intent, gate, car, story, notes, reused } = readTrace(events);
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
        <span className="cost-badge" title="How many times the AI was asked. Most steps need no AI at all.">
          {llm.length === 0 ? "No AI" : `${llm.length} AI`}
        </span>
      </div>

      {car && <div className="pipeline-sub">{car}</div>}

      <div className="stages">
        {PIPELINE.map((stage) => {
          const state = stateOf(stage.key);
          const called = stage.agent && llm.includes(stage.agent);
          return (
            <div key={stage.key} className={`stage ${state}`}>
              <StatusDot state={state} />
              <span className="stage-label">
                {stage.label}
                {called && <span className="agent-tag on">AI</span>}
                {stage.agent && !called && state === "done" && (
                  <span className="agent-tag saved">no AI</span>
                )}
              </span>
            </div>
          );
        })}
      </div>

      <div className="pipeline-details">
        <div className="detail-steps">
          {PIPELINE.map((stage) => (
            <div key={stage.key} className={`detail-row ${stateOf(stage.key)}`}>
              <div className="detail-step">{stage.label}</div>
              <div className="detail-text">{detailOf(stage)}</div>
            </div>
          ))}
        </div>

        {gate && (
          <div className={`gate-card ${gate.hit ? "hit" : "miss"}`}>
            <div className="gate-verdict">
              {gate.hit ? "We've fixed this before" : "New problem for us"}
            </div>
            <div className="gate-detail">
              {gate.hit
                ? "Answered from a repair that was confirmed on a car like yours, so the AI wasn't asked to diagnose."
                : gate.tier === "loose"
                  ? "The closest past repair is on a different model, so it was passed to the AI as a hint rather than used as the answer."
                  : "Nothing matching in our past repairs, so this went to the AI for a full diagnosis."}
            </div>
          </div>
        )}

        {lookups.length > 0 && (
          <div className="lookups">
            {lookups.map((l) => (
              <div key={l.name} className={`lookup ${l.count === 0 ? "empty" : ""}`}>
                <span className="source">
                  {SOURCE_LABEL[l.name] || l.name}
                  {l.calls > 1 && <em> · searched {l.calls}×</em>}
                </span>
                <span className="count">
                  {l.count === 0 ? "nothing found" : `${l.count} found`}
                </span>
                {l.name.startsWith("research.") && l.count === 0 && (
                  <span className="stub-tag">not connected yet</span>
                )}
              </div>
            ))}
          </div>
        )}

        {story.length > 0 && (
          <ul className="story">
            {story.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        )}

        {notes.length > 0 && (
          <details className="trace-log">
            <summary>Technical details</summary>
            <pre>{notes.join("\n")}</pre>
          </details>
        )}
      </div>
    </div>
  );
}

// `latest` is the last answer in the conversation. Only it carries the urgency
// verdict: "do not drive" is advice about the car as it stands now, and three
// of them stacked down the thread -- two of them superseded -- is three
// conflicting instructions rather than a history.
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
              {c.evidence.length > 0 && (
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
function hasCard(answer) {
  return !(answer?.message || "").trim();
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

function ClipIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
      <path
        d="M8 12l6.5-6.5a3 3 0 1 1 4.24 4.24L10 18.5a5 5 0 1 1-7.07-7.07L11.5 3"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
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
  const [imageType, setImageType] = useState("vin_plate");
  // Keyed by request id rather than a single slot: switching records used to
  // wipe the run, so an open pipeline vanished and could only be brought back
  // by re-running the diagnosis. Each record now keeps its own.
  const [runs, setRuns] = useState({});
  // Which records have their steps expanded. Never cleared -- an open panel
  // stays open for the rest of the session, however much you click around.
  const [openSteps, setOpenSteps] = useState(() => new Set());
  const fileRef = useRef(null);
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

  const uploadImage = async (file) => {
    if (!file || !activeId) return;
    const body = new FormData();
    body.append("request", activeId);
    body.append("image_type", imageType);
    body.append("image", file);
    await api("/api/diagnostic-images/", { method: "POST", body });
    await loadActive(activeId);
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
                  {hasRun && (
                    <button
                      type="button"
                      className={`req-dots ${open ? "open" : ""}`}
                      aria-expanded={open}
                      title={open ? "Hide the steps" : "Show the steps behind this answer"}
                      onClick={(e) => {
                        e.stopPropagation(); // the row itself only selects
                        toggleSteps(r.id);
                      }}
                    >
                      ···
                    </button>
                  )}
                </div>
                {open && <Pipeline run={rowRun} />}
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
                <label>What's going on?</label>
                <textarea
                  required
                  value={form.raw_text}
                  onChange={(e) => setForm({ ...form, raw_text: e.target.value })}
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
              </span>
              <span className={`pill ${active.status}`}>{STATUS_LABEL[active.status] || active.status}</span>

              <select
                className="attach-select"
                value={imageType}
                onChange={(e) => setImageType(e.target.value)}
                title="Photo type for next upload"
              >
                {IMAGE_TYPES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
              <input
                type="file"
                ref={fileRef}
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => uploadImage(e.target.files[0])}
              />
              <button className="icon-btn" title="Attach photo" onClick={() => fileRef.current?.click()}>
                <ClipIcon />
              </button>
              <button
                className="primary-btn compact"
                onClick={() => runPipeline(activeId)}
                disabled={activeRun.running}
                title="Re-run the agentic pipeline over the conversation so far"
              >
                {activeRun.running ? "Running…" : "Run diagnosis"}
              </button>
            </div>

            {active.images.length > 0 && (
              <div className="images">
                {active.images.map((img) => (
                  <img key={img.id} src={img.image} title={img.image_type} alt={img.image_type} />
                ))}
              </div>
            )}

            {active.case_matches.length > 0 && (
              <div className="match-card">
                <div className="match-title">Similar cases found in the knowledge base</div>
                {active.case_matches.map((m) => (
                  <div key={m.id} className="match-row">
                    <span className="confidence">{Math.round(m.confidence * 100)}%</span> — {m.case.symptom_text} →{" "}
                    {m.case.confirmed_fix}
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

                {/* The new answer lands at the bottom when the run finishes.
                    Until then this holds its place -- showing the previous
                    answer here would read as a reply to the question above it. */}
                {(sending || activeRun.running) && (
                  <div className="answer-pending">
                    <span className="pending-dots">
                      <i />
                      <i />
                      <i />
                    </span>
                    Working out a new answer…
                  </div>
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
    </div>
  );
}
