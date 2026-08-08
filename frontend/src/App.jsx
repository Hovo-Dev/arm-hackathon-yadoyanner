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
  const fileRef = useRef(null);
  const messagesRef = useRef(null);

  const loadRequests = () => api("/api/diagnostics/").then((data) => setRequests(data.results));
  const loadActive = (id) => api(`/api/diagnostics/${id}/`).then((data) => setActive(data));

  useEffect(() => {
    loadRequests().catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    if (activeId) loadActive(activeId).catch((err) => console.error(err));
    else setActive(null);
  }, [activeId]);

  useEffect(() => {
    if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [active?.messages]);

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
    } finally {
      setCreating(false);
    }
  };

  const sendMessage = async () => {
    const content = chatInput.trim();
    if (!content || !activeId || sending) return;
    setChatInput("");
    setSending(true);

    const assistantId = uid();
    setActive((prev) => ({
      ...prev,
      messages: [
        ...prev.messages,
        { id: uid(), role: "user", content },
        { id: assistantId, role: "assistant", content: "" },
      ],
    }));

    const setAssistantText = (text) => {
      setActive((prev) => ({
        ...prev,
        messages: prev.messages.map((m) => (m.id === assistantId ? { ...m, content: text } : m)),
      }));
    };

    try {
      const res = await fetch(`/api/diagnostics/${activeId}/messages/stream/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      if (!res.ok || !res.body) throw new Error(`${res.status} ${await res.text()}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let text = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const line = rawEvent.replace(/^data:\s*/, "");
          if (!line) continue;
          const payload = JSON.parse(line);
          if (payload.delta) {
            text += payload.delta;
            setAssistantText(text);
          }
        }
      }
    } catch (err) {
      setAssistantText(`[error: ${err.message}]`);
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
          {requests.map((r) => (
            <div
              key={r.id}
              className={`req-item ${r.id === activeId ? "active" : ""}`}
              onClick={() => setActiveId(r.id)}
            >
              <span className={`dot ${r.status}`} />
              <span className="title">
                {r.car_make} {r.car_model} {r.car_year || ""}
              </span>
            </div>
          ))}
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
                  value={form.raw_text}
                  onChange={(e) => setForm({ ...form, raw_text: e.target.value })}
                  placeholder="Grinding noise when braking at low speed..."
                />
                <button className="primary-btn" type="submit" disabled={creating}>
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
                {active.messages.map((m) => (
                  <div key={m.id} className={`message-row ${m.role}`}>
                    {m.role === "assistant" && (
                      <div className="avatar bot">
                        <LogoMark size={16} />
                      </div>
                    )}
                    <div className={`bubble ${m.role}`}>{m.content}</div>
                    {m.role === "user" && <div className="avatar user">You</div>}
                  </div>
                ))}
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
                  placeholder="Ask a follow-up question..."
                  rows={1}
                />
                <button className="send-btn" onClick={sendMessage} disabled={sending}>
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
