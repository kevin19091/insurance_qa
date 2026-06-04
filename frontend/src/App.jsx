import { useState, useRef, useCallback } from "react";
import Markdown from "react-markdown";
import "./App.css";

export default function App() {
  const [query, setQuery] = useState("");
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [abortController, setAbortController] = useState(null);
  const inputRef = useRef(null);

  const handleSubmit = useCallback(
    (e) => {
      e.preventDefault();
      const q = query.trim();
      if (!q || loading) return;

      const question = { role: "user", content: q };
      const answer = { role: "assistant", content: "", sources: [], streaming: true };
      setConversation((prev) => [...prev, question, answer]);
      setQuery("");
      setLoading(true);

      const url = `/api/chat/stream?q=${encodeURIComponent(q)}`;
      const eventSource = new EventSource(url);

      eventSource.addEventListener("sources", (e) => {
        const data = JSON.parse(e.data);
        setConversation((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            last.sources = data.sources;
          }
          return updated;
        });
      });

      eventSource.addEventListener("token", (e) => {
        const data = JSON.parse(e.data);
        setConversation((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            last.content += data.token;
          }
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
      });
    },
    [query, loading],
  );

  return (
    <div className="app">
      <header className="header">
        <h1>Insurance QnA Bot</h1>
        <p className="subtitle">Ask questions about your policy document</p>
      </header>

      <main className="chat">
        {conversation.length === 0 && (
          <div className="empty-state">
            <p>Ask a question about the Max Life Group Credit Life Secure Plan.</p>
          </div>
        )}

        {conversation.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="avatar">{msg.role === "user" ? "You" : "Bot"}</div>
            <div className="bubble">
              {msg.role === "assistant" && msg.content ? (
                <Markdown>{msg.content}</Markdown>
              ) : (
                <p>{msg.content}</p>
              )}

              {msg.role === "assistant" && msg.streaming && (
                <span className="cursor" />
              )}

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
          This response is AI-generated and does not constitute legal or
          insurance advice. Always refer to your policy document for definitive
          information.
        </p>
      </footer>
    </div>
  );
}
