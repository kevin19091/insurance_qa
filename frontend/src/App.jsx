import { useState, useRef, useCallback, useEffect } from "react";
import Markdown from "react-markdown";
import "./App.css";

export default function App() {
  const [query, setQuery] = useState("");
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [strategies, setStrategies] = useState(null);
  const [overrides, setOverrides] = useState({});

  const eventSourceRef = useRef(null);
  const inputRef = useRef(null);
  const chatRef = useRef(null);

  useEffect(() => {
    fetch("/api/strategies")
      .then((r) => r.json())
      .then(setStrategies)
      .catch(() => {});
  }, []);

  const buildUrl = useCallback(
    (q) => {
      const params = new URLSearchParams({ q });
      if (devMode) {
        params.set("mode", "dev");
        for (const [key, value] of Object.entries(overrides)) {
          if (value !== undefined && value !== null && value !== "") {
            params.set(key, value);
          }
        }
      }
      return `/api/chat/stream?${params.toString()}`;
    },
    [devMode, overrides],
  );

  const handleSubmit = useCallback(
    (e) => {
      e.preventDefault();
      const q = query.trim();
      if (!q || loading) return;

      const question = { role: "user", content: q };
      const answer = { role: "assistant", content: "", sources: [], steps: [], streaming: true, pipelineTrace: null };
      setConversation((prev) => [...prev, question, answer]);
      setQuery("");
      setLoading(true);

      const url = buildUrl(q);
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.addEventListener("step", (e) => {
        const data = JSON.parse(e.data);
        if (data.status === "start") {
          setConversation((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant" && last.steps) {
              last.steps = [...last.steps, { step: data.step, status: "running", duration_ms: 0, cost_usd: 0 }];
            }
            return updated;
          });
        } else {
          setConversation((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant" && last.steps) {
              last.steps = last.steps.map((s) =>
                s.step === data.step ? { step: data.step, status: "complete", duration_ms: data.duration_ms, cost_usd: data.cost_usd } : s,
              );
            }
            return updated;
          });
        }
      });

      eventSource.addEventListener("sources", (e) => {
        const data = JSON.parse(e.data);
        setConversation((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") last.sources = data.sources;
          return updated;
        });
      });

      eventSource.addEventListener("token", (e) => {
        const data = JSON.parse(e.data);
        setConversation((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") last.content += data.token;
          return updated;
        });
      });

      eventSource.addEventListener("pipeline_trace", (e) => {
        const data = JSON.parse(e.data);
        setConversation((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") last.pipelineTrace = data.steps;
          return updated;
        });
      });

      eventSource.addEventListener("done", () => {
        setConversation((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") last.streaming = false;
          return updated;
        });
        setLoading(false);
        eventSource.close();
        eventSourceRef.current = null;
        chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
      });

      eventSource.addEventListener("error", (e) => {
        try {
          const data = JSON.parse(e.data || "{}");
          setConversation((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              last.content = data.error || "An error occurred.";
              last.streaming = false;
            }
            return updated;
          });
        } catch {
          // ignore parse errors from EventSource connection errors
        }
        setLoading(false);
        eventSource.close();
        eventSourceRef.current = null;
      });
    },
    [query, loading, buildUrl],
  );

  const updateOverride = (key, value) => {
    setOverrides((prev) => ({ ...prev, [key]: value }));
  };

  const totalCost = (steps) => {
    if (!steps) return 0;
    return steps.reduce((sum, s) => sum + (s.cost_usd || 0), 0);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-row">
          <h1>Insurance QnA Bot</h1>
          <label className="dev-toggle">
            <input type="checkbox" checked={devMode} onChange={(e) => setDevMode(e.target.checked)} />
            <span>Dev Mode</span>
          </label>
        </div>
        <p className="subtitle">Ask questions about your policy document</p>
      </header>

      {devMode && strategies && (
        <div className="dev-panel">
          <div className="dev-controls">
            {Object.entries(strategies).map(([key, strat]) => {
              if (strat.overridable) {
                const paramKey = key === "top_k" ? "top_k" : key === "retrieval" ? "retrieval_mode" : key === "query_rewrite" ? "rewrite_strategy" : key === "llm" ? "llm_model" : key === "reranker" ? "reranker_model" : key;
                return (
                  <label key={key} className="control-group">
                    <span>{key.replace(/_/g, " ")}</span>
                    <select value={overrides[paramKey] || ""} onChange={(e) => updateOverride(paramKey, e.target.value)}>
                      <option value="">Default ({strat.default})</option>
                      {strat.options.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  </label>
                );
              }
              return (
                <label key={key} className="control-group readonly">
                  <span>{key.replace(/_/g, " ")}</span>
                  <span className="readonly-value">{strat.default}</span>
                </label>
              );
            })}
          </div>
        </div>
      )}

      <main className="chat" ref={chatRef}>
        {conversation.length === 0 && (
          <div className="empty-state">
            <p>Ask a question about the Max Life Group Credit Life Secure Plan.</p>
          </div>
        )}

        {conversation.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="avatar">{msg.role === "user" ? "You" : "Bot"}</div>
            <div className="bubble">
              {msg.role === "assistant" && msg.steps?.length > 0 && devMode && (
                <div className="pipeline-steps">
                  {msg.steps.map((step, j) => (
                    <div key={j} className={`step ${step.status}`}>
                      <span className="step-name">{step.step}</span>
                      {step.status === "running" && <span className="step-spinner" />}
                      {step.status === "complete" && (
                        <span className="step-metrics">
                          {step.duration_ms.toFixed(0)}ms
                          {step.cost_usd > 0 && ` · $${step.cost_usd.toFixed(4)}`}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {msg.role === "assistant" && msg.content ? (
                <Markdown>{msg.content}</Markdown>
              ) : (
                <p>{msg.content}</p>
              )}

              {msg.role === "assistant" && msg.streaming && <span className="cursor" />}

              {msg.role === "assistant" && msg.sources?.length > 0 && !msg.streaming && (
                <details className="sources">
                  <summary>Sources ({msg.sources.length})</summary>
                  {msg.sources.map((s, j) => (
                    <div key={j} className="source-item">
                      <span className="page">Page {s.page}</span>
                      <span className="score">Score: {s.score}</span>
                      <p className="preview">{s.text}</p>
                    </div>
                  ))}
                </details>
              )}

              {msg.role === "assistant" && msg.pipelineTrace && devMode && !msg.streaming && (
                <details className="trace-summary">
                  <summary>Pipeline Summary</summary>
                  <div className="trace-steps">
                    {msg.pipelineTrace.map((step, j) => (
                      <div key={j} className="trace-row">
                        <span className="trace-step">{step.step}</span>
                        <span className="trace-duration">{step.duration_ms.toFixed(0)}ms</span>
                        {step.cost_usd > 0 && <span className="trace-cost">${step.cost_usd.toFixed(4)}</span>}
                      </div>
                    ))}
                  </div>
                  <div className="trace-total">
                    Total: {msg.pipelineTrace.reduce((s, st) => s + st.duration_ms, 0).toFixed(0)}ms · ${totalCost(msg.pipelineTrace).toFixed(4)}
                  </div>
                </details>
              )}
            </div>
          </div>
        ))}
      </main>

      <form className="input-bar" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your policy..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "..." : "Send"}
        </button>
      </form>

      <footer className="disclaimer">
        <p>
          This response is AI-generated and does not constitute legal or insurance advice.
          Always refer to your policy document for definitive information.
        </p>
      </footer>
    </div>
  );
}
