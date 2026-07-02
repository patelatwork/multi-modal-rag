# Multimodal Chat-to-PDF

This project ingests PDFs with Unstructured, summarizes text, tables, and images with an OpenAI-compatible multimodal model, and stores the results in a persistent Chroma vector store plus a persistent LangChain docstore for MultiVectorRetriever lookup.

## Setup

1. Install the Python dependencies from requirements.txt.
2. Install the external PDF helpers required by Unstructured: poppler-utils and tesseract-ocr.
3. Add the environment variables below to .env.

## Environment variables

- LLM_BASE_URL: OpenAI-compatible endpoint URL such as OpenRouter, Fireworks, or Ollama.
- LLM_API_KEY: API token for that endpoint. For local Ollama, any placeholder string is fine.
- LLM_MODEL: Defaults to meta/llama-4-scout-17b-16e-instruct.
- HF_API_TOKEN: Hugging Face token used for the embedding endpoint.
- HF_EMBEDDING_MODEL: Defaults to sentence-transformers/all-MiniLM-L6-v2.
- HF_API_URL: Optional Hugging Face inference endpoint URL.

Example .env values:

LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your_openrouter_key
LLM_MODEL=meta/llama-4-scout-17b-16e-instruct
HF_API_TOKEN=your_hugging_face_token
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

## Run

python run.py path/to/document.pdf

Add --question "your question" to run a single QA pass after indexing.

## Persistence

- Vector embeddings are stored in storage/chroma.
- Raw extracted documents are stored in storage/docstore.
- Both locations are reused on later runs, so the index is persistent.
