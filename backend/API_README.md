# Backend API Reference

This backend is a FastAPI service for:
- Local Ollama chat via `POST /api/chat` (optional RAG; streaming or JSON)
- The same `POST /api/chat` with `cloud: true` for OpenAI (GPT) or Anthropic (Opus/Sonnet) using server-side API keys
- Standalone cloud streaming via `POST /api/cloud/chat` (OpenAI, Anthropic, Google; API key in the request body)
- Model management (list, load, stop, loaded list)
- RAG knowledge-base ingestion and chunk management
- Basic health and system info

Base URL (default):
- `http://localhost:8000`

Default Ollama chat model when `model` is omitted on `POST /api/chat` comes from **`backend/settings.json`** → **`default_chat_model`** (default value **`gpt-oss:20b`** if the key is missing or the file is absent). Restart the server after editing the file.

## Quick Start

From the `backend` directory:

```bash
./run.sh
```

Health check:

```bash
curl http://localhost:8000/api/health
```

### `settings.json` (backend)

Path: `backend/settings.json` (same directory as `main.py`).

```json
{
  "default_chat_model": "gpt-oss:20b"
}
```

---

## Environment variables and `.env` files

On startup, the server loads (in order):

1. **`.env`** from the project root, then from `backend/` (values do not override variables already set in the process environment).
2. **`.env.local`** from the project root, then `backend/`, then `frontend/` (each file may override keys loaded earlier from `.env`).

Useful variables:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Used when `POST /api/chat` is sent with `"cloud": true` and the model name indicates OpenAI (see below). |
| `ANTHROPIC_API_KEY` | Used when `POST /api/chat` is sent with `"cloud": true` and the model name indicates Anthropic (see below). |
| `OLLAMA_KEEP_ALIVE` | How long Ollama keeps chat models loaded between requests (default indefinite). |
| `RAG_EMBEDDING_MODEL` | Ollama embedding model id for RAG (default `embeddinggemma:latest`). |
| `RAG_EMBEDDING_KEEP_ALIVE` | Ollama keep-alive for the embedding model (default `0` so embeddings are not pinned in VRAM between RAG calls). |
| `MAX_CONTEXT_MESSAGES` | Max messages kept after trimming (default `30`). |

Example `backend/.env.local` (or project root / `frontend/.env.local`):

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Do not commit real keys; keep `.env.local` out of version control.

---

## Chat Endpoints

### `POST /api/chat`
Chat with optional **local Ollama** (default) or **cloud** (OpenAI / Anthropic) on the same endpoint. RAG context from the vector store is applied in both modes when retrieval runs.

- Content-Type: `application/json`
- Response:
  - **`stream: true` (default):** streamed **plain text** body (`text/plain`)
  - **`stream: false`:** single JSON object with the full assistant message

Request body:

```json
{
  "model": "gpt-oss:20b",
  "messages": [
    { "role": "user", "content": "Explain recursion simply." }
  ],
  "thread_id": "optional-thread-id",
  "instruction": "Optional. Prepended to the start of the last user message for this request (affects model input and RAG query).",
  "stream": true,
  "cloud": false
}
```

#### `cloud` (default `false`) and automatic cloud routing

The server uses **OpenAI or Anthropic** (with **`OPENAI_API_KEY`** / **`ANTHROPIC_API_KEY`**) when **either**:

- **`cloud` is `true`**, or  
- **`model`** looks like a cloud API id (inference), so clients can omit `cloud` for OpenAI-style ids such as **`gpt-5.4`** or **`gpt-4o`**.

Inference rules (substring / prefix checks on a lowercased `model`):

- **Local Ollama:** ids containing **`gpt-oss`** are always treated as Ollama, not OpenAI.
- **Anthropic:** **`opus`**, **`sonnet`**, **`haiku`**, or **`claude`** in the name.
- **OpenAI:** id **`gpt-…`** (e.g. `gpt-4o`, `gpt-5.4`) or **`o1` / `o3` / `o4`…** prefixes (e.g. `o3-mini`).

If none of the above match and **`cloud` is `false`**, the request uses **local Ollama** (same as before). If **`cloud` is `true`** but the model id does not map to a provider, the server returns **`400`**.

When **`cloud` is `false`** and the model is inferred as cloud (e.g. `gpt-5.4`), Ollama load checks are skipped and the OpenAI API is used instead.

Cloud chat uses the same RAG pipeline as local chat (retrieval + merged system context), then calls the provider’s chat API. Streaming uses the same wire format as local chat (`text/plain` chunks).

**Local Ollama notes:**

- Only **`default_chat_model`** may be used without an explicit **Load** in Manage / `POST /api/models/load`. Any other Ollama model must already be running (`ollama ps`); otherwise the server returns **`400`**.
- If `model` equals the configured **`RAG_EMBEDDING_MODEL`**, chat returns **`400`** (embedding models are not valid chat models).

**Cloud notes:**

- If the required API key is missing, the server returns **`503`**.
- Non-stream JSON responses include extra fields: `"cloud": true` and `"provider": "openai"` or `"anthropic"`.

Streaming example:

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [{"role":"user","content":"What is dynamic programming?"}],
    "thread_id": "thread-1"
  }'
```

Non-streaming example (one JSON response):

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [{"role":"user","content":"Say hello in one sentence."}],
    "stream": false
  }'
```

Example response when `stream` is `false` (local):

```json
{
  "message": { "role": "assistant", "content": "Hello — nice to meet you." },
  "model": "gpt-oss:20b"
}
```

Example response when `stream` is `false` (cloud):

```json
{
  "message": { "role": "assistant", "content": "Hello — nice to meet you." },
  "model": "gpt-4o",
  "cloud": true,
  "provider": "openai"
}
```

Cloud streaming example (requires `OPENAI_API_KEY`):

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "cloud": true,
    "model": "gpt-4o",
    "messages": [{"role":"user","content":"One sentence about FastAPI."}]
  }'
```

Notes:
- `model` is optional in JSON for **local** chat; if omitted, the server uses **`default_chat_model`** from **`settings.json`** next to `main.py` (falls back to **`gpt-oss:20b`**). For **`cloud: true`**, set `model` to a real OpenAI or Anthropic model id.
- `thread_id` is optional and used for short-term in-memory thread history.
- `instruction` is optional; it is applied **after** context trimming, to the **last** `user` message only (prefix + blank line + original content).
- Message context is trimmed to `MAX_CONTEXT_MESSAGES` (default 30).

---

### `GET /api/chat/threads/{thread_id}`
Return stored short-term messages for a thread.

Example:

```bash
curl http://localhost:8000/api/chat/threads/thread-1
```

---

### `POST /api/cloud/chat`
Stream chat from a cloud provider. Unlike `POST /api/chat` with `cloud: true`, this endpoint expects the **API key in the JSON body** (no `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` required on the server).

- Supported providers: `openai`, `anthropic`, `google`
- Content-Type: `application/json`
- Response: streamed plain text

Request body:

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "messages": [
    { "role": "user", "content": "Say hello." }
  ],
  "api_key": "YOUR_API_KEY"
}
```

Example:

```bash
curl -N -X POST http://localhost:8000/api/cloud/chat \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "messages": [{"role":"user","content":"Give me 3 bullet points about caching."}],
    "api_key": "YOUR_API_KEY"
  }'
```

---

## Model Management Endpoints

### `GET /api/models`
List locally available Ollama models.

```bash
curl http://localhost:8000/api/models
```

Response shape:

```json
{ "models": ["gpt-oss:20b", "gemma3:12b"] }
```

---

### `GET /api/models/loaded`
List currently loaded/running Ollama models.

```bash
curl http://localhost:8000/api/models/loaded
```

---

### `POST /api/models/load`
Load a model into memory.

Request body:

```json
{ "model": "gpt-oss:20b" }
```

Example:

```bash
curl -X POST http://localhost:8000/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss:20b"}'
```

---

### `POST /api/models/stop`
Stop/unload a model from memory.

Request body:

```json
{ "model": "gpt-oss:20b" }
```

Example:

```bash
curl -X POST http://localhost:8000/api/models/stop \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss:20b"}'
```

---

## RAG Endpoints

### `POST /api/rag/documents`
Add raw text documents to the vector store (chunk + embed).

Request body:

```json
{
  "texts": [
    "First text block...",
    "Second text block..."
  ]
}
```

Example:

```bash
curl -X POST http://localhost:8000/api/rag/documents \
  -H "Content-Type: application/json" \
  -d '{"texts":["FastAPI is a modern Python web framework."]}'
```

---

### `POST /api/rag/documents/files`
Upload `.md` and `.txt` files for ingestion.

- Content-Type: `multipart/form-data`
- Field name: `files`
- Multiple files supported

Example:

```bash
curl -X POST http://localhost:8000/api/rag/documents/files \
  -F "files=@/absolute/path/notes.md" \
  -F "files=@/absolute/path/todo.txt"
```

---

### `GET /api/rag/status`
Return embedding model, collection, storage path, and chunk count.

```bash
curl http://localhost:8000/api/rag/status
```

---

### `GET /api/rag/chunks?limit=500&offset=0`
List stored chunks with preview and metadata.

```bash
curl "http://localhost:8000/api/rag/chunks?limit=100&offset=0"
```

---

### `DELETE /api/rag/chunks`
Delete specific chunks by IDs.

Request body:

```json
{
  "ids": ["chunk-id-1", "chunk-id-2"]
}
```

Example:

```bash
curl -X DELETE http://localhost:8000/api/rag/chunks \
  -H "Content-Type: application/json" \
  -d '{"ids":["chunk-id-1","chunk-id-2"]}'
```

---

### `DELETE /api/rag/storage`
Clear all vector storage/chunks.

```bash
curl -X DELETE http://localhost:8000/api/rag/storage
```

---

## Utility Endpoints

### `GET /api/health`
Basic service health check.

```bash
curl http://localhost:8000/api/health
```

---

### `GET /api/sysinfo`
Return system memory stats.

```bash
curl http://localhost:8000/api/sysinfo
```

---

## Common Error Cases

- `400`: Invalid input (empty payload, unknown provider, `cloud: true` with a model name that is neither GPT nor Opus/Sonnet, Ollama model not loaded when it is not the default, embedding model used for chat, etc.)
- `500`: Internal/backend dependency errors
- `502`: Upstream OpenAI/Anthropic error body (non-stream cloud completions on `POST /api/chat`)
- `503`: Missing RAG dependencies when RAG endpoints are used; missing **`OPENAI_API_KEY`** or **`ANTHROPIC_API_KEY`** when **`POST /api/chat`** is called with **`"cloud": true`**

For RAG file upload, ensure:
- `python-multipart` is installed
- uploaded files are `.md` or `.txt`

