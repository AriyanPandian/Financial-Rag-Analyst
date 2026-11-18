# LedgerLens — Financial RAG Analyst

Upload a financial filing and ask questions against its extracted text. The application stores vectors locally in Chroma, uses OpenAI for embeddings and answer generation, and returns source excerpts with PDF page numbers or HTML section headings.

## Features

- Extracts text from text-based PDF filings, SEC-style HTML, Markdown, and UTF-8 text.
- Preserves PDF page numbers, HTML section headings, and table rows for citations.
- Chunks and embeds documents, persists vectors in Chroma, and tracks document metadata in SQLite.
- Answers questions from retrieved excerpts and only returns citations that match retrieved source labels.
- Supports document listing/removal, upload size limits, optional bearer-token protection, and a responsive browser interface.

Scanned/image-only PDFs need OCR before upload. This app is designed for a trusted, single-tenant deployment; set an API access token and put it behind HTTPS and an identity-aware reverse proxy before exposing it beyond localhost.

## Run locally

Requirements: Python 3.11+ and an OpenAI API key.

```bash
cd "Financial Rag Analyst"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the server:

```bash
set -a
source .env
set +a
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Indexed files and vectors are persisted under `DATA_DIR` (default: `./data`).

## Docker

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, then run:

```bash
docker compose up --build
```

The service binds to localhost on port 8000 and persists its data in a Docker volume. Configure `API_ACCESS_TOKEN` to protect API operations; enter the same token in the UI. `/api/health` remains public and only reports whether the provider is configured.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Required for indexing and chat |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat completion model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `DATA_DIR` | `./data` | Persistent Chroma and SQLite data |
| `MAX_UPLOAD_MB` | `50` | Maximum upload size |
| `API_ACCESS_TOKEN` | unset | Optional bearer token for API operations |

## API

- `GET /api/health` — readiness/configuration status
- `GET /api/documents` — indexed document metadata
- `POST /api/documents` — multipart upload (`file`)
- `DELETE /api/documents/{id}` — remove a document and its vectors
- `POST /api/chat` — ask a question with optional recent conversation history

Interactive API documentation is available at `/docs`.

## Tests

```bash
python -m unittest discover -s tests -v
```
