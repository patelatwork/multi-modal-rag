import { useCallback, useEffect, useRef, useState } from "react";
import { api, waitForJob } from "./api";
import { Message } from "./components/Message";
import { Sidebar } from "./components/Sidebar";
import type { ChatMessage, DocumentSummary, Health, Job } from "./types";

const SUGGESTIONS = [
  "Which model performs best, and by how much?",
  "Summarise what the figures show.",
  "Reproduce the results table.",
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [job, setJob] = useState<Job | null>(null);

  const scrollAnchor = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [docs, status] = await Promise.all([api.documents(), api.health()]);
      setDocuments(docs);
      setHealth(status);
    } catch {
      /* the sidebar simply stays stale if the API is down */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  const handleUpload = useCallback(
    async (file: File) => {
      try {
        const started = await api.upload(file);
        setJob(started);
        const finished = await waitForJob(started.job_id, setJob);
        setJob(finished);
        if (finished.status === "succeeded") {
          await refresh();
          // Clear the banner once the document shows up in the list.
          setTimeout(() => setJob(null), 1200);
        }
      } catch (error) {
        setJob({
          job_id: "local",
          status: "failed",
          filename: file.name,
          processed: 0,
          total: 0,
          result: null,
          error: error instanceof Error ? error.message : "Upload failed",
        });
      }
    },
    [refresh],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      await api.deleteDocument(id);
      setSelected((current) => current.filter((value) => value !== id));
      await refresh();
    },
    [refresh],
  );

  const submit = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || asking) return;

      const userMessage: ChatMessage = {
        id: `u-${Date.now()}`,
        role: "user",
        content: trimmed,
      };
      setMessages((current) => [...current, userMessage]);
      setQuestion("");
      setAsking(true);

      try {
        const response = await api.ask(trimmed, selected);
        setMessages((current) => [
          ...current,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: response.answer,
            sources: response.sources,
            meta: {
              images: response.images_sent_to_model,
              seconds: response.elapsed_seconds,
            },
          },
        ]);
      } catch (error) {
        setMessages((current) => [
          ...current,
          {
            id: `e-${Date.now()}`,
            role: "assistant",
            content:
              error instanceof Error ? error.message : "Something went wrong.",
            error: true,
          },
        ]);
      } finally {
        setAsking(false);
        composer.current?.focus();
      }
    },
    [asking, selected],
  );

  const empty = messages.length === 0;

  return (
    <div className="app">
      <Sidebar
        health={health}
        documents={documents}
        selected={selected}
        job={job}
        onToggle={(id) =>
          setSelected((current) =>
            current.includes(id)
              ? current.filter((value) => value !== id)
              : [...current, id],
          )
        }
        onUpload={handleUpload}
        onDelete={handleDelete}
      />

      <main className="chat">
        <div className="transcript">
          {empty && (
            <div className="welcome">
              <h2>Ask your documents anything</h2>
              <p>
                Charts and tables are rendered as images and read by a vision
                model, so answers can cite what a chart actually shows — not
                just the text around it.
              </p>
              {documents.length === 0 ? (
                <p className="hint">Upload a PDF to get started.</p>
              ) : (
                <div className="suggestions">
                  {SUGGESTIONS.map((suggestion) => (
                    <button key={suggestion} onClick={() => void submit(suggestion)}>
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}

          {asking && (
            <div className="message assistant">
              <div className="bubble thinking">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
          <div ref={scrollAnchor} />
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void submit(question);
          }}
        >
          <textarea
            ref={composer}
            rows={1}
            value={question}
            placeholder={
              documents.length === 0
                ? "Upload a PDF first…"
                : "Ask about the documents…"
            }
            disabled={documents.length === 0}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends; Shift+Enter inserts a newline.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submit(question);
              }
            }}
          />
          <button type="submit" disabled={asking || !question.trim()}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
