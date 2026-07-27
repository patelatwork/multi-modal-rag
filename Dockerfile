# Multi-stage: the frontend is built with Node, then served as static files by
# the same FastAPI process, so a deployment is a single container.

FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Bake the embedding model into the image so the first request is not a
    # cold download.
    HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY multimodal_rag/ ./multimodal_rag/
COPY run.py pyproject.toml ./
COPY --from=frontend /frontend/dist ./static

# Data lives on a volume; the image stays stateless.
ENV INPUT_PDF_DIR=/data/pdfs \
    EXTRACTION_DIR=/data/extracted \
    VECTOR_STORE_DIR=/data/chroma \
    DOCSTORE_DIR=/data/docstore \
    API_HOST=0.0.0.0
VOLUME ["/data"]

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data && chown -R app:app /data /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["python", "-m", "uvicorn", "multimodal_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
