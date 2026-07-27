import type { ChatResponse, DocumentSummary, Health, Job } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) {
    // The API returns {detail, kind}; fall back to the status line for
    // anything else (a proxy error, say).
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),

  documents: () => request<DocumentSummary[]>("/api/documents"),

  deleteDocument: (id: string) =>
    request<void>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),

  ask: (question: string, documentIds?: string[]) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        document_ids: documentIds?.length ? documentIds : null,
      }),
    }),

  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Job>("/api/documents", { method: "POST", body: form });
  },

  job: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`),
};

/**
 * Poll an ingestion job until it settles.
 *
 * Ingestion makes one vision call per figure, so a long PDF can run for
 * minutes; polling keeps the connection free and lets the UI show progress.
 */
export async function waitForJob(
  jobId: string,
  onProgress: (job: Job) => void,
  intervalMs = 1500,
): Promise<Job> {
  for (;;) {
    const job = await api.job(jobId);
    onProgress(job);
    if (job.status === "succeeded" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

export { ApiError };
