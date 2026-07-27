import { useRef } from "react";
import type { DocumentSummary, Health, Job } from "../types";

interface Props {
  health: Health | null;
  documents: DocumentSummary[];
  selected: string[];
  job: Job | null;
  onToggle: (id: string) => void;
  onUpload: (file: File) => void;
  onDelete: (id: string) => void;
}

export function Sidebar({
  health,
  documents,
  selected,
  job,
  onToggle,
  onUpload,
  onDelete,
}: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const busy = job?.status === "pending" || job?.status === "running";

  return (
    <aside className="sidebar">
      <div className="brand">
        <h1>Multimodal RAG</h1>
        <p>Ask across text, tables and figures.</p>
      </div>

      <input
        ref={fileInput}
        type="file"
        accept="application/pdf,.pdf"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onUpload(file);
          event.target.value = "";
        }}
      />
      <button className="upload" disabled={busy} onClick={() => fileInput.current?.click()}>
        {busy ? "Ingesting…" : "＋ Upload PDF"}
      </button>

      {job && busy && (
        <div className="job">
          <div className="job-name">{job.filename}</div>
          <div className="progress">
            <div
              className="progress-bar"
              style={{ width: job.total ? `${(100 * job.processed) / job.total}%` : "15%" }}
            />
          </div>
          <div className="job-detail">
            {job.total ? `summarising ${job.processed}/${job.total}` : "extracting…"}
          </div>
        </div>
      )}
      {job?.status === "failed" && <div className="job-error">{job.error}</div>}

      <div className="doc-list">
        <div className="section-title">
          Documents
          {selected.length > 0 && <span className="filter-note">{selected.length} filtered</span>}
        </div>

        {documents.length === 0 && <p className="empty">Nothing indexed yet.</p>}

        {documents.map((doc) => (
          <div
            key={doc.document_id}
            className={`doc${selected.includes(doc.document_id) ? " selected" : ""}`}
          >
            <button className="doc-main" onClick={() => onToggle(doc.document_id)}>
              <span className="doc-name" title={doc.filename}>
                {doc.filename}
              </span>
              <span className="doc-meta">
                {doc.pages} pages · {doc.element_count} elements
              </span>
            </button>
            <button
              className="doc-delete"
              title="Remove from index"
              onClick={() => onDelete(doc.document_id)}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {health && (
        <div className="health">
          <span className={`dot ${health.status}`} />
          <span>{health.chat_model}</span>
          <span className="health-sub">
            {health.documents} docs · {health.elements} elements
          </span>
          {health.detail && <span className="health-warn">{health.detail}</span>}
        </div>
      )}
    </aside>
  );
}
