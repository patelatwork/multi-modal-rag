export type ElementKind = "text" | "table" | "figure";

export interface Source {
  id: string;
  label: string;
  document_id: string;
  kind: ElementKind;
  page_number: number | null;
  caption: string | null;
  text: string;
  score: number | null;
  image_url: string | null;
  table_markdown: string | null;
}

export interface ChatResponse {
  question: string;
  answer: string;
  sources: Source[];
  images_sent_to_model: number;
  elapsed_seconds: number;
}

export interface DocumentSummary {
  document_id: string;
  filename: string;
  pages: number;
  element_count: number;
  backend: string | null;
  created_at: string;
}

export interface IngestionResult {
  document_id: string;
  filename: string;
  pages: number;
  text_chunks: number;
  tables: number;
  figures: number;
  skipped_existing: boolean;
  elapsed_seconds: number;
  warnings: string[];
}

export interface Job {
  job_id: string;
  status: "pending" | "running" | "succeeded" | "failed";
  filename: string;
  processed: number;
  total: number;
  result: IngestionResult | null;
  error: string | null;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  chat_model: string;
  embedding_provider: string;
  extractor_backend: string;
  documents: number;
  elements: number;
  detail: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  meta?: { images: number; seconds: number };
  error?: boolean;
}
