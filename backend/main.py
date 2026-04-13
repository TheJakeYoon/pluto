import ollama
import os
import html
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Annotated, Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Query, Path
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from collections import OrderedDict
from contextlib import asynccontextmanager
import psutil
import subprocess
import httpx
import json
import time

# Local Ollama and cloud OpenAI chat use LangChain (ChatOllama / ChatOpenAI) for one stack.
# Ollama: https://docs.langchain.com/oss/python/integrations/chat/ollama
# OpenAI: https://docs.langchain.com/oss/python/integrations/chat/openai
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from logging_setup import configure_app_logging
from ollama_structured import ollama_structured_chat_complete
from structured_chat import anthropic_structured_complete, openai_structured_complete

# ChromaDB persist directory under project root / database
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_BASE_DIR, "settings.json")
_DB_DIR = os.path.join(_BASE_DIR, "..", "database", "chroma")
os.makedirs(_DB_DIR, exist_ok=True)

_PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))
for _env_dir in (_PROJECT_ROOT, _BASE_DIR):
    _env_base = os.path.join(_env_dir, ".env")
    if os.path.isfile(_env_base):
        load_dotenv(_env_base, override=False)
for _env_dir in (_PROJECT_ROOT, _BASE_DIR, os.path.join(_PROJECT_ROOT, "frontend")):
    _env_local = os.path.join(_env_dir, ".env.local")
    if os.path.isfile(_env_local):
        load_dotenv(_env_local, override=True)


def _load_backend_settings() -> dict:
    """Load backend/settings.json. Missing or invalid file → empty dict (see ``_default_chat_model_from_settings``)."""
    try:
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print("Warning: settings.json must be a JSON object; ignoring.")
            return {}
        return dict(data)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse settings.json ({e}); ignoring.")
        return {}


_backend_settings = _load_backend_settings()


def _default_chat_model_from_settings(settings: dict) -> str:
    v = settings.get("default_chat_model")
    if isinstance(v, str) and v.strip():
        return v.strip()
    lm = settings.get("local_models")
    if isinstance(lm, list) and lm:
        first = lm[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return "gpt-oss:20b"

# RAG: embedding model for vector store (Ollama). Use one already loaded, e.g. embeddinggemma:latest
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "embeddinggemma:latest")
RAG_COLLECTION = "chatbot_kb"
RAG_TOP_K = 4

# Keep chat models resident between requests (-1 = indefinite). Override with OLLAMA_KEEP_ALIVE if needed.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "-1")
try:
    OLLAMA_KEEP_ALIVE = int(OLLAMA_KEEP_ALIVE)
except ValueError:
    pass  # e.g. "5m" string

# RAG embeddings: default 0 so the embedding model is not kept loaded (only default_chat + manually loaded models stay resident).
# Set RAG_EMBEDDING_KEEP_ALIVE=-1 to pin the embedding model like chat models.
_RAG_EMBED_KA = os.environ.get("RAG_EMBEDDING_KEEP_ALIVE", "0")
try:
    RAG_EMBEDDING_KEEP_ALIVE = int(_RAG_EMBED_KA)
except ValueError:
    RAG_EMBEDDING_KEEP_ALIVE = _RAG_EMBED_KA  # e.g. "5m"

# Default Ollama chat model when the client omits `model` (settings.json → default_chat_model, else local_models[0]).
DEFAULT_CHAT_MODEL = _default_chat_model_from_settings(_backend_settings)

_embeddings = None
_vector_store = None
_rag_import_error = None  # set if lazy RAG imports fail


def _rag_imports():
    """Lazy import for RAG/Chroma. Raises HTTPException if deps are missing (e.g. not using venv)."""
    global _rag_import_error
    if _rag_import_error is not None:
        raise HTTPException(
            status_code=503,
            detail="RAG dependencies not available. Run the backend with the venv: cd backend && ./venv/bin/uvicorn main:app --reload",
        )
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_core.documents import Document
        from langchain_chroma import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return OllamaEmbeddings, Document, Chroma, RecursiveCharacterTextSplitter
    except ImportError as e:
        _rag_import_error = e
        raise HTTPException(
            status_code=503,
            detail="RAG dependencies not installed. Activate the backend venv and run: pip install -r requirements.txt (then restart with ./venv/bin/uvicorn main:app --reload)",
        )


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        OllamaEmbeddings, _, _, _ = _rag_imports()
        _embeddings = OllamaEmbeddings(model=RAG_EMBEDDING_MODEL, keep_alive=RAG_EMBEDDING_KEEP_ALIVE)
    return _embeddings


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _, _, Chroma, _ = _rag_imports()
        _vector_store = Chroma(
            collection_name=RAG_COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=_DB_DIR,
        )
    return _vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    """No Ollama preload: models stay unloaded until the UI or ``POST /api/models/load``."""
    _log = logging.getLogger("uvicorn.error")
    _log.info(
        "Ollama: skipping startup preload. Default chat model id for omitted `model` field: %s",
        DEFAULT_CHAT_MODEL,
    )
    yield

app = FastAPI(
    title="Ollama Chatbot API",
    lifespan=lifespan,
    description=(
        "Local Ollama chat with optional RAG, plus cloud chat (OpenAI / Anthropic via server env keys, "
        "Google via client-supplied key). See the human-readable guide at /api/documentation or /api/doc."
    ),
)

configure_app_logging(_BASE_DIR)


def _configure_access_log_filter() -> None:
    """Hide high-frequency polling routes from uvicorn access logs (not model load events)."""

    class _QuietPollPathsFilter(logging.Filter):
        _NEEDLES = (
            "GET /api/models/loaded ",
            "GET /api/models HTTP",
            "GET /api/sysinfo ",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return not any(n in msg for n in self._NEEDLES)

    logging.getLogger("uvicorn.access").addFilter(_QuietPollPathsFilter())


_configure_access_log_filter()

# Do not log custom lines on `uvicorn.access`: its formatter expects record.args (client, method, …).
# Use `uvicorn.error` so messages appear in the same console as uvicorn (always configured).
_api_timestamp_logger = logging.getLogger("uvicorn.error")


def _format_http_exception_detail_for_log(detail: object) -> str:
    if detail is None:
        return "(no detail)"
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(detail)


def _log_upstream_http_error_visible(request: Request, exc: HTTPException) -> None:
    """High-visibility log for bad-gateway / upstream failures (console + rotating file)."""
    detail_text = _format_http_exception_detail_for_log(exc.detail)
    client = request.client.host if request.client else "?"
    bar = "=" * 78
    msg = (
        "\n"
        + bar
        + "\n"
        + f"  HTTP {exc.status_code}  {request.method} {request.url.path}\n"
        + f"  client={client}\n"
        + bar
        + "\n"
        + detail_text
        + "\n"
        + bar
    )
    _api_timestamp_logger.error("%s", msg)


@app.exception_handler(HTTPException)
async def _http_exception_upstream_visible_log(request: Request, exc: HTTPException):
    if exc.status_code in (502, 503):
        _log_upstream_http_error_visible(request, exc)
    return await http_exception_handler(request, exc)


# Avoid buffering huge request bodies in middleware (multipart / large uploads).
_HTTP_SKIP_BODY_PREFIX = ("multipart/", "application/octet-stream")
_HTTP_MAX_PREFETCH = int(os.environ.get("HTTP_MIDDLEWARE_MAX_BODY_BYTES", str(2 * 1024 * 1024)))


def _local_iso_timestamp() -> str:
    """Return local timezone timestamp in ISO-8601 format."""
    return datetime.now().astimezone().isoformat()


_SENSITIVE_REQUEST_HEADER_NAMES = frozenset(
    {"authorization", "cookie", "x-api-key", "proxy-authorization"}
)


def _request_headers_for_bad_request_log(request: Request) -> dict:
    """Header names/values suitable for logs (credentials redacted)."""
    out: dict = {}
    for key, value in request.headers.items():
        lk = key.lower()
        out[key] = "[REDACTED]" if lk in _SENSITIVE_REQUEST_HEADER_NAMES else value
    return out


def _body_value_for_bad_request_log(body: bytes, content_type: str) -> object:
    """Parse JSON when possible; otherwise UTF-8 text with a size cap."""
    if not body:
        return None
    ct = (content_type or "").lower()
    stripped = body.lstrip()
    if "application/json" in ct or (stripped[:1] in (b"{", b"[")):
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    text = body.decode("utf-8", errors="replace")
    max_chars = 100_000
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
    return text


def _bad_request_request_dump(request: Request, body_cache: Optional[bytes]) -> str:
    payload = {
        "method": request.method,
        "path": request.url.path,
        "url": str(request.url),
        "client_host": request.client.host if request.client else None,
        "query": dict(request.query_params),
        "headers": _request_headers_for_bad_request_log(request),
        "body": (
            _body_value_for_bad_request_log(
                body_cache, request.headers.get("content-type", "")
            )
            if body_cache is not None
            else None
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


@app.middleware("http")
async def add_local_timestamps(request: Request, call_next):
    req_ts = _local_iso_timestamp()
    request.state.request_local_ts = req_ts
    _api_timestamp_logger.info(
        "[local_ts=%s] Incoming %s %s", req_ts, request.method, request.url.path
    )

    body_cache: Optional[bytes] = None
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        ct = (request.headers.get("content-type") or "").lower()
        cl_raw = request.headers.get("content-length")
        skip_body_cache = any(ct.startswith(p) for p in _HTTP_SKIP_BODY_PREFIX)
        if not skip_body_cache and cl_raw:
            try:
                if int(cl_raw) > _HTTP_MAX_PREFETCH:
                    skip_body_cache = True
            except ValueError:
                pass
        if not skip_body_cache:
            body_cache = await request.body()

            async def receive():
                return {"type": "http.request", "body": body_cache, "more_body": False}

            request._receive = receive

    response = await call_next(request)
    resp_ts = _local_iso_timestamp()
    response.headers["X-Request-Local-Time"] = req_ts
    response.headers["X-Response-Local-Time"] = resp_ts
    _api_timestamp_logger.info(
        "[local_ts=%s] Response %s %s %s",
        resp_ts,
        response.status_code,
        request.method,
        request.url.path,
    )
    if response.status_code == 400:
        dump = _bad_request_request_dump(request, body_cache)
        _api_timestamp_logger.warning(
            "[local_ts=%s] HTTP 400 bad request — request (human-readable JSON):\n%s",
            resp_ts,
            dump,
        )
    return response

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    """One turn in the chat history."""

    role: str = Field(
        ...,
        description='Message author: typically "user", "assistant", or "system".',
    )
    content: str = Field(..., description="Plain-text (or markdown) message body.")


OpenAIReasoningEffort = Optional[Literal["none", "low", "medium", "high", "xhigh"]]


def _openai_reasoning_effort_optional(val: OpenAIReasoningEffort) -> Optional[str]:
    """Map API field to LangChain ``ChatOpenAI.reasoning_effort`` (None → omit parameter)."""
    if val is None or val == "none":
        return None
    return val


class StructuredToolDefinition(BaseModel):
    """Custom Ollama tool schema when structured_output is true (local models only)."""

    name: str = Field(..., description="Function name the model must call.")
    description: str = Field("", description="Short description shown to the model.")
    parameters: Dict[str, Any] = Field(
        ...,
        description='JSON Schema for arguments, e.g. {"type":"object","properties":{...},"required":[...]}.',
    )


class ChatRequest(BaseModel):
    """POST /api/chat JSON body."""

    model: str = Field(
        DEFAULT_CHAT_MODEL,
        description=(
            "Model id. Local: Ollama name (e.g. gpt-oss:20b). Cloud: OpenAI (gpt-…) or Anthropic (claude-…) id; "
            "provider is inferred from the id unless cloud=true."
        ),
    )
    messages: List[ChatMessage] = Field(
        ...,
        description="Full conversation for this request (server trims to last MAX_CONTEXT_MESSAGES).",
    )
    thread_id: Optional[str] = Field(
        None,
        description="If set, server appends this exchange to in-memory history for that id (GET /api/chat/threads/{thread_id}).",
    )
    instruction: Optional[str] = Field(
        None,
        description="Optional text prepended to the last user message for this call only (system-style nudge).",
    )
    stream: bool = Field(
        True,
        description="If true, response is text/plain streamed tokens. If false, JSON object with full assistant message.",
    )
    cloud: bool = Field(
        False,
        description="If true, use cloud LLM with OPENAI_API_KEY or ANTHROPIC_API_KEY from server env / .env.local.",
    )
    thinking: bool = Field(
        True,
        description=(
            "Ollama (local): passed to LangChain ``ChatOllama`` as ``reasoning`` for non-Gemma models (enables "
            "Ollama thinking/reasoning API). Gemma 4: when true, prepends <|think|> on the RAG system message and "
            "streamed replies strip hidden thought; structured_output uses ``reasoning=false`` only when thinking is false."
        ),
    )
    reasoning: OpenAIReasoningEffort = Field(
        None,
        description=(
            "OpenAI (cloud only): maps to LangChain ``ChatOpenAI.reasoning_effort`` for reasoning models "
            "(omit when null or ``none``; ``low`` | ``medium`` | ``high`` | ``xhigh`` sent as-is)."
        ),
    )
    structured_output: Optional[bool] = Field(
        default=None,
        description=(
            "Optional; omit or null for false. When true with stream=false, use LangChain structured output (forced tool / function calling): "
            "local Ollama (https://docs.ollama.com/capabilities/tool-calling) or cloud OpenAI/Anthropic. "
            "Response includes structured; with local Ollama set structured_tool for a custom JSON shape."
        ),
    )
    structured_tool: Optional[StructuredToolDefinition] = Field(
        None,
        description=(
            "Optional custom tool (name + JSON Schema parameters) for local Ollama when structured_output is true. "
            "If omitted, the default deliver_chat_response tool (markdown/math segments) is used. Not supported on cloud."
        ),
    )

class CloudChatRequest(BaseModel):
    """POST /api/cloud/chat — browser-key providers (e.g. Google)."""

    provider: str = Field(
        ...,
        description='Cloud id: "openai", "anthropic", or "google".',
    )
    model: str = Field(..., description="Provider-specific model id.")
    messages: List[ChatMessage] = Field(..., description="Chat history for the request.")
    api_key: str = Field(..., description="API key for this provider (sent from client; not used for server-key OpenAI/Anthropic on /api/chat).")
    reasoning: OpenAIReasoningEffort = Field(
        None,
        description="OpenAI only: same values as POST /api/chat ``reasoning`` → LangChain ``ChatOpenAI.reasoning_effort``.",
    )


class AddDocumentsRequest(BaseModel):
    """Add texts to the RAG knowledge base. Each string is split into chunks and embedded."""

    texts: List[str] = Field(
        ...,
        description="Non-empty strings to chunk, embed, and add to Chroma (embedding model from RAG_EMBEDDING_MODEL).",
    )


class DeleteChunksRequest(BaseModel):
    """Delete specific chunks from the vector store by their IDs."""

    ids: List[str] = Field(..., description="Chroma chunk ids to delete (from GET /api/rag/chunks).")


# Short-term memory: keep last N messages so context stays focused and within model limits
# See: https://docs.langchain.com/oss/python/langchain/short-term-memory
MAX_CONTEXT_MESSAGES = int(os.environ.get("MAX_CONTEXT_MESSAGES", "30"))


def _list_available_models() -> List[str]:
    """Return locally available Ollama models, preferring `ollama ls` output."""
    try:
        result = subprocess.run(
            ["ollama", "ls"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return []
        # Expected table format: NAME ID SIZE MODIFIED
        models: List[str] = []
        for line in lines[1:]:
            if not line:
                continue
            name = line.split()[0]
            if name and name not in models:
                models.append(name)
        if models:
            return models
    except Exception:
        pass

    # Fallback to SDK listing if CLI parsing fails.
    client = ollama.Client()
    model_list = client.list()
    if hasattr(model_list, "models"):
        return [m.model for m in model_list.models]
    if isinstance(model_list, dict):
        return [m.get("name") or m.get("model") for m in model_list.get("models", []) if (m.get("name") or m.get("model"))]
    return []


def _ollama_model_labels_equivalent(a: str, b: str) -> bool:
    """True if two Ollama model names refer to the same model (tag optional on either side)."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    ba, _, ta = a.partition(":")
    bb, _, tb = b.partition(":")
    if ba != bb:
        return False
    if not ta or not tb:
        return True
    return ta == tb


async def _async_ollama_loaded_model_names() -> List[str]:
    client = ollama.AsyncClient()
    try:
        model_list = await client.ps()
        if hasattr(model_list, "models"):
            return [m.model for m in model_list.models]
        if isinstance(model_list, dict):
            return [
                m.get("name") or m.get("model")
                for m in model_list.get("models", [])
                if (m.get("name") or m.get("model"))
            ]
        return []
    except Exception:
        return []


async def _ollama_chat_model_is_loaded(model: str) -> bool:
    loaded = await _async_ollama_loaded_model_names()
    return any(_ollama_model_labels_equivalent(model, x) for x in loaded)


def _trim_messages(messages: List[dict], max_messages: int = MAX_CONTEXT_MESSAGES) -> List[dict]:
    """Keep the first system message (if any) and the last max_messages-1 messages for context."""
    if len(messages) <= max_messages:
        return messages
    first = []
    if messages and (messages[0].get("role") or "").lower() == "system":
        first = [messages[0]]
        max_rest = max_messages - 1
    else:
        max_rest = max_messages
    return first + messages[-max_rest:]


# In-memory thread store: bounded so the process can run indefinitely.
THREAD_STORE_MAX_THREADS = int(os.environ.get("THREAD_STORE_MAX_THREADS", "5000"))
THREAD_STORE_MAX_MESSAGES = int(os.environ.get("THREAD_STORE_MAX_MESSAGES", str(MAX_CONTEXT_MESSAGES)))
_thread_store: OrderedDict[str, List[dict]] = OrderedDict()


def _thread_store_put(thread_id: str, messages: List[dict]) -> None:
    """Trim messages per thread and LRU-evict least-recently-used threads when over cap."""
    trimmed = _trim_messages(messages, max_messages=THREAD_STORE_MAX_MESSAGES)
    _thread_store[thread_id] = trimmed
    _thread_store.move_to_end(thread_id)
    while len(_thread_store) > THREAD_STORE_MAX_THREADS:
        _thread_store.popitem(last=False)


def _to_langchain_messages(messages: List[dict]) -> List:
    """Convert API messages to LangChain message types (see ChatOllama integration)."""
    lc_messages = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


GEMMA4_THINK_TRIGGER = "<|think|>"
# Gemma 4 model output: reasoning between these markers, then the user-facing answer (see Ollama Gemma 4 readme).
GEMMA4_THOUGHT_START = "<|channel>thought"
GEMMA4_THOUGHT_END = "<channel|>"


def _is_gemma4_model(model: str) -> bool:
    return "gemma4" in (model or "").lower()


def _ollama_reasoning_invoke_kw(*, thinking: bool, model: str) -> dict[str, Any]:
    """Map ``thinking`` to LangChain ``ChatOllama`` ``reasoning`` (Gemma 4 uses ``<|think|>`` on the system prompt instead)."""
    if _is_gemma4_model(model):
        return {}
    return {"reasoning": bool(thinking)}


def _ollama_chat_sampling_kwargs(model: str) -> dict:
    """Gemma 4 on Ollama recommends temperature=1, top_p=0.95, top_k=64; other models keep deterministic defaults."""
    if _is_gemma4_model(model):
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    return {"temperature": 0}


def _gemma4_wall_think_answer_from_raw(raw: str, wall_total_s: float) -> tuple[float, float]:
    """Approximate wall time spent in thought channel vs public answer by Gemma 4 marker positions."""
    if not raw or wall_total_s <= 0:
        return 0.0, wall_total_s
    si = raw.find(GEMMA4_THOUGHT_START)
    if si == -1:
        return 0.0, wall_total_s
    ei = raw.find(GEMMA4_THOUGHT_END, si + len(GEMMA4_THOUGHT_START))
    if ei == -1:
        return wall_total_s, 0.0
    after = ei + len(GEMMA4_THOUGHT_END)
    thought_len = max(0, after)
    answer_len = max(0, len(raw) - after)
    denom = max(1, thought_len + answer_len)
    return wall_total_s * thought_len / denom, wall_total_s * answer_len / denom


def _gemma4_eval_think_answer_from_raw(raw: str, eval_ns: Optional[int]) -> tuple[Optional[float], Optional[float]]:
    """Split Ollama eval_duration (ns) between thought and answer using character lengths inside markers."""
    if not isinstance(eval_ns, int) or eval_ns <= 0 or not raw:
        return None, None
    si = raw.find(GEMMA4_THOUGHT_START)
    if si == -1:
        return None, float(eval_ns) / 1e9
    ei = raw.find(GEMMA4_THOUGHT_END, si + len(GEMMA4_THOUGHT_START))
    if ei == -1:
        return float(eval_ns) / 1e9, None
    after = ei + len(GEMMA4_THOUGHT_END)
    thought_len = max(0, after)
    answer_len = max(0, len(raw) - after)
    denom = max(1, thought_len + answer_len)
    return (eval_ns * thought_len / denom) / 1e9, (eval_ns * answer_len / denom) / 1e9


def _log_ollama_chat_finished(
    *,
    model: str,
    thinking_enabled: bool,
    streaming: bool,
    wall_total_s: float,
    wall_think_s: Optional[float],
    wall_answer_s: Optional[float],
    ollama_eval_think_s: Optional[float],
    ollama_eval_answer_s: Optional[float],
    response_metadata: Optional[dict],
) -> None:
    parts = [
        f"model={model!r}",
        f"streaming={streaming}",
        f"thinking_request={thinking_enabled}",
        f"wall_total_s={wall_total_s:.3f}",
    ]
    if wall_think_s is not None:
        parts.append(f"wall_think_s={wall_think_s:.3f}")
    if wall_answer_s is not None:
        parts.append(f"wall_answer_s={wall_answer_s:.3f}")
    if ollama_eval_think_s is not None:
        parts.append(f"ollama_eval_think_s={ollama_eval_think_s:.3f}")
    if ollama_eval_answer_s is not None:
        parts.append(f"ollama_eval_answer_s={ollama_eval_answer_s:.3f}")
    rm = response_metadata or {}
    for key in ("total_duration", "eval_duration", "load_duration", "prompt_eval_duration"):
        ns = rm.get(key)
        if isinstance(ns, (int, float)) and ns > 0:
            parts.append(f"ollama_{key}_ms={ns / 1e6:.2f}")
    _api_timestamp_logger.info("Ollama chat finished: %s", " ".join(parts))


def _strip_gemma4_public_answer(text: str) -> str:
    """Remove Gemma 4 internal thought blocks; keep only content after the closing marker."""
    if not text:
        return text
    s = text
    while True:
        i = s.find(GEMMA4_THOUGHT_START)
        if i == -1:
            break
        j = s.find(GEMMA4_THOUGHT_END, i + len(GEMMA4_THOUGHT_START))
        if j == -1:
            return (s[:i]).rstrip()
        s = s[:i] + s[j + len(GEMMA4_THOUGHT_END) :]
    return s


async def _stream_strip_gemma4_thought(
    achunk_stream,
    *,
    timing: Optional[dict] = None,
    ollama_meta: Optional[list] = None,
):
    """Buffer until the first thought block closes, then stream deltas (public answer only)."""
    buf = ""
    past = False
    raw_parts: list[str] = []

    def _note_first_public() -> None:
        if timing is not None and "t_first_public" not in timing:
            timing["t_first_public"] = time.perf_counter()

    async for chunk in achunk_stream:
        rm = getattr(chunk, "response_metadata", None) or {}
        if rm and ollama_meta is not None:
            ollama_meta[0] = rm
        c = getattr(chunk, "content", None) or ""
        raw_parts.append(c)
        if not c:
            continue
        if past:
            _note_first_public()
            yield c
            continue
        buf += c
        si = buf.find(GEMMA4_THOUGHT_START)
        if si != -1:
            ei = buf.find(GEMMA4_THOUGHT_END, si + len(GEMMA4_THOUGHT_START))
            if ei != -1:
                past = True
                tail = buf[ei + len(GEMMA4_THOUGHT_END) :]
                buf = ""
                if tail:
                    _note_first_public()
                    yield tail
    if not past and buf:
        _note_first_public()
        yield _strip_gemma4_public_answer(buf)
    if timing is not None:
        timing["raw_model_output"] = "".join(raw_parts)


def _apply_instruction_to_last_user(messages: List[dict], instruction: Optional[str]) -> List[dict]:
    """Prepend optional instruction to the beginning of the last user message (model input for this turn)."""
    if not instruction or not str(instruction).strip():
        return messages
    inst = str(instruction).strip()
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if (out[i].get("role") or "").lower() == "user":
            base = out[i].get("content") or ""
            out[i] = {**out[i], "content": f"{inst}\n\n{base}" if base else inst}
            break
    return out


def _build_ollama_lc_messages_with_rag(
    messages: List[dict], *, thinking: bool = False, model: str = ""
) -> List:
    """RAG retrieve from latest user text, then LangChain messages + RAG system prefix."""
    last_user_content = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user_content = messages[i]["content"]
            break
    context_parts = []
    query = (last_user_content or "").strip() or "general"
    try:
        vs = _get_vector_store()
        retriever = vs.as_retriever(search_kwargs={"k": RAG_TOP_K})
        docs = retriever.invoke(query)
        if docs:
            context_parts = [doc.page_content for doc in docs]
    except HTTPException:
        raise
    except Exception:
        pass
    if context_parts:
        rag_system = (
            "You must treat the following context as the only source of truth. "
            "Base your answer strictly on this context. Do not add, infer, or state information that is not in the context. "
            "If the context does not contain enough information to fully answer the question, say so and answer only from what is given. "
            "Do not contradict or go beyond the context.\n\n"
            "Context:\n" + "\n\n---\n\n".join(context_parts)
        )
    else:
        rag_system = (
            "No relevant context from the knowledge base was found for this query. "
            "Answer from your general knowledge."
        )
    lc_messages = _to_langchain_messages(messages)
    first_system = SystemMessage(content=rag_system)
    if thinking and _is_gemma4_model(model):
        first_system = SystemMessage(content=f"{GEMMA4_THINK_TRIGGER}\n{rag_system}")
        _api_timestamp_logger.info(
            "Gemma 4 thinking enabled for model=%s: <|think|> prepended to RAG system message.",
            model,
        )
    return [first_system] + lc_messages


async def _ollama_chat_stream_chunks(
    model: str,
    messages: List[dict],
    *,
    thinking: bool = False,
    timing: Optional[dict] = None,
):
    """Yields text tokens from ChatOllama (RAG + streaming).

    If ``timing`` is a dict, it is filled with: ``t_start``, ``t_end``, ``t_first_public`` (perf_counter
    values), and ``ollama_response_metadata`` (last chunk's ``response_metadata`` from LangChain).
    """
    lc_messages = _build_ollama_lc_messages_with_rag(messages, thinking=thinking, model=model)
    llm = ChatOllama(
        model=model,
        keep_alive=OLLAMA_KEEP_ALIVE,
        **_ollama_chat_sampling_kwargs(model),
    )
    stream = llm.astream(lc_messages, **_ollama_reasoning_invoke_kw(thinking=thinking, model=model))
    t_start = time.perf_counter()
    if timing is not None:
        timing["t_start"] = t_start
    ollama_meta: list = [None]
    if _is_gemma4_model(model):
        async for part in _stream_strip_gemma4_thought(stream, timing=timing, ollama_meta=ollama_meta):
            yield part
    else:
        raw_parts: list[str] = []
        async for chunk in stream:
            rm = getattr(chunk, "response_metadata", None) or {}
            if rm:
                ollama_meta[0] = rm
            raw_parts.append(chunk.content or "")
            if chunk.content:
                if timing is not None and "t_first_public" not in timing:
                    timing["t_first_public"] = time.perf_counter()
                yield chunk.content
        if timing is not None:
            timing["raw_model_output"] = "".join(raw_parts)
    if timing is not None:
        timing["t_end"] = time.perf_counter()
        timing["ollama_response_metadata"] = ollama_meta[0]


def _assistant_content_from_invoke(result) -> str:
    c = getattr(result, "content", result)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(c) if c is not None else ""


async def _ollama_chat_complete_text(model: str, messages: List[dict], *, thinking: bool = False) -> str:
    """Single non-streaming completion (same RAG path as streaming)."""
    lc_messages = _build_ollama_lc_messages_with_rag(messages, thinking=thinking, model=model)
    llm = ChatOllama(
        model=model,
        keep_alive=OLLAMA_KEEP_ALIVE,
        **_ollama_chat_sampling_kwargs(model),
    )
    t0 = time.perf_counter()
    result = await llm.ainvoke(lc_messages, **_ollama_reasoning_invoke_kw(thinking=thinking, model=model))
    t1 = time.perf_counter()
    raw = _assistant_content_from_invoke(result)
    rm = getattr(result, "response_metadata", None) or {}
    eval_ns = rm.get("eval_duration")
    if _is_gemma4_model(model):
        wall_think, wall_answer = _gemma4_wall_think_answer_from_raw(raw, t1 - t0)
        ev_think, ev_answer = _gemma4_eval_think_answer_from_raw(raw, eval_ns if isinstance(eval_ns, int) else None)
    else:
        wall_think, wall_answer = None, t1 - t0
        ev_think, ev_answer = None, None
    _log_ollama_chat_finished(
        model=model,
        thinking_enabled=thinking,
        streaming=False,
        wall_total_s=t1 - t0,
        wall_think_s=wall_think,
        wall_answer_s=wall_answer,
        ollama_eval_think_s=ev_think,
        ollama_eval_answer_s=ev_answer,
        response_metadata=rm,
    )
    if _is_gemma4_model(model):
        return _strip_gemma4_public_answer(raw)
    return raw


def _langchain_to_openai_compatible_dicts(lc_messages: List) -> List[dict]:
    """Convert LangChain messages to OpenAI/Anthropic-style role/content dicts."""
    out: List[dict] = []
    for m in lc_messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
    return out


def _collapse_system_messages(messages: List[dict]) -> List[dict]:
    """Merge multiple system messages into one (cloud APIs expect a single system block)."""
    systems: List[str] = []
    rest: List[dict] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "system":
            c = m.get("content")
            if c is not None and str(c).strip():
                systems.append(str(c))
        else:
            rest.append(m)
    if not systems:
        return rest
    combined = "\n\n".join(systems)
    return [{"role": "system", "content": combined}] + rest


def _messages_for_cloud_chat_with_rag(
    messages: List[dict],
    *,
    thinking: bool = False,
    model: str = "",
) -> List[dict]:
    """Same RAG context as local Ollama chat, formatted for OpenAI/Anthropic."""
    lc = _build_ollama_lc_messages_with_rag(messages, thinking=thinking, model=model)
    flat = _langchain_to_openai_compatible_dicts(lc)
    return _collapse_system_messages(flat)


def _infer_cloud_provider_from_model_id(model: str) -> Optional[str]:
    """Map API model ids to cloud provider. None means treat as local Ollama.

    Excludes ``gpt-oss`` (Ollama) from OpenAI routing. OpenAI API models typically
    start with ``gpt``, ``o1``, ``o3``, or ``o4``. Anthropic ids usually contain
    claude/opus/sonnet/haiku.
    """
    m = (model or "").lower().strip()
    if not m:
        return None
    if "gpt-oss" in m:
        return None
    if "opus" in m or "sonnet" in m or "haiku" in m or "claude" in m:
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    return None


def _cloud_provider_for_model(model: str) -> str:
    """Resolve provider when cloud=true but model id is ambiguous for inference."""
    inferred = _infer_cloud_provider_from_model_id(model)
    if inferred is not None:
        return inferred
    raise HTTPException(
        status_code=400,
        detail=(
            "When cloud=true, use an OpenAI model id (e.g. gpt-4o, gpt-5.4), "
            "an Anthropic Claude id (e.g. claude-sonnet-…), or set cloud=false for Ollama."
        ),
    )


def _chat_api_messages_to_langchain(messages: List[ChatMessage]) -> list:
    out: list = []
    for m in messages:
        role = (m.role or "user").lower()
        if role == "system":
            out.append(SystemMessage(content=m.content))
        elif role == "assistant":
            out.append(AIMessage(content=m.content))
        else:
            out.append(HumanMessage(content=m.content))
    return out


async def _openai_chat_complete_messages(
    model: str,
    messages: List[ChatMessage],
    api_key: str,
    *,
    reasoning_effort: Optional[str] = None,
) -> str:
    lc = _chat_api_messages_to_langchain(messages)
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            timeout=httpx.Timeout(30.0, read=300.0),
            max_retries=2,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
        )
        result = await llm.ainvoke(lc)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return _assistant_content_from_invoke(result)


async def _anthropic_chat_complete_messages(model: str, messages: List[ChatMessage], api_key: str) -> str:
    system_text = ""
    msgs: List[dict] = []
    for m in messages:
        if m.role == "system":
            system_text = (system_text + "\n\n" + m.content).strip() if system_text else m.content
        else:
            msgs.append({"role": m.role, "content": m.content})
    payload: dict = {"model": model, "max_tokens": 4096, "messages": msgs}
    if system_text:
        payload["system"] = system_text
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=httpx.Timeout(30.0, read=300.0),
        )
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=r.text)
        data = r.json()
        blocks = data.get("content") or []
        return "".join(
            b.get("text", "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )


@app.post(
    "/api/chat",
    summary="Chat completion",
    response_description=(
        "If stream=true: text/plain body streamed incrementally. "
        "If stream=false: JSON with message and model; if structured_output, structured "
        "(segments for default tool, or tool_name/arguments for Ollama structured_tool)."
    ),
)
async def chat_completion(request: ChatRequest):
    """Run chat against **local Ollama** or **cloud OpenAI/Anthropic** (server API keys).

    **Routing:** If `model` looks like a cloud id (e.g. `gpt-4o`, `claude-sonnet-4-6`) or `cloud` is true,
    the request uses env keys `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. Otherwise Ollama is used; non-default
    models must already be loaded (`POST /api/models/load`).

    **structured_output** with `stream=false`: uses LangChain ``with_structured_output`` (function calling) —
    local Ollama (see ollama.com tool calling) or cloud OpenAI/Anthropic. Default tool is `deliver_chat_response` (segments); set **structured_tool**
    for a custom JSON schema. **structured_tool** is not supported on cloud.

    **thinking** (Ollama) and **reasoning** (OpenAI cloud): see field descriptions; both map into LangChain chat models.
    """
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    messages = _trim_messages(messages)
    messages = _apply_instruction_to_last_user(messages, request.instruction)
    thread_id = request.thread_id

    want_structured_output = bool(request.structured_output)

    inferred_cloud = _infer_cloud_provider_from_model_id(request.model)
    use_cloud_llm = bool(request.cloud) or (inferred_cloud is not None)

    if use_cloud_llm:
        if request.structured_tool is not None:
            raise HTTPException(
                status_code=400,
                detail="structured_tool is only supported with local Ollama (use cloud=false and a local model id).",
            )
        provider = inferred_cloud if inferred_cloud is not None else _cloud_provider_for_model(request.model)
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            env_key_name = "OPENAI_API_KEY"
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            env_key_name = "ANTHROPIC_API_KEY"
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"cloud=true requires {env_key_name} in the environment or in .env.local "
                    f"(project root or backend/)."
                ),
            )
        try:
            cloud_msgs = _messages_for_cloud_chat_with_rag(
                messages,
                thinking=bool(request.thinking),
                model=request.model,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        chat_messages = [ChatMessage(role=m["role"], content=str(m["content"])) for m in cloud_msgs]
        cloud_req = CloudChatRequest(
            provider=provider,
            model=request.model,
            messages=chat_messages,
            api_key=api_key,
            reasoning=request.reasoning,
        )

        if want_structured_output and request.stream:
            raise HTTPException(
                status_code=400,
                detail="structured_output requires stream=false (tool calls are not used on the streaming path).",
            )

        if not request.stream:
            try:
                if want_structured_output:
                    cloud_dicts = [{"role": m.role, "content": m.content} for m in chat_messages]
                    if provider == "openai":
                        full_content, segments = await openai_structured_complete(
                            request.model,
                            cloud_dicts,
                            api_key,
                            reasoning_effort=_openai_reasoning_effort_optional(request.reasoning),
                        )
                    else:
                        full_content, segments = await anthropic_structured_complete(
                            request.model, cloud_dicts, api_key
                        )
                    if thread_id:
                        stored = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                        stored.append({"role": "assistant", "content": full_content})
                        _thread_store_put(thread_id, stored)
                    return {
                        "message": {"role": "assistant", "content": full_content},
                        "structured": {"segments": segments},
                        "model": request.model,
                        "cloud": True,
                        "provider": provider,
                    }
                if provider == "openai":
                    full_content = await _openai_chat_complete_messages(
                        request.model,
                        chat_messages,
                        api_key,
                        reasoning_effort=_openai_reasoning_effort_optional(request.reasoning),
                    )
                else:
                    full_content = await _anthropic_chat_complete_messages(
                        request.model, chat_messages, api_key
                    )
                if thread_id:
                    stored = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                    stored.append({"role": "assistant", "content": full_content})
                    _thread_store_put(thread_id, stored)
                return {
                    "message": {"role": "assistant", "content": full_content},
                    "model": request.model,
                    "cloud": True,
                    "provider": provider,
                }
            except HTTPException:
                raise
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Structured tool response could not be parsed: {e}",
                )
            except RuntimeError as e:
                raise HTTPException(status_code=502, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        streamer = CLOUD_STREAMERS[provider]

        async def generate_cloud():
            full_content = ""
            try:
                async for part in streamer(cloud_req):
                    full_content += part
                    yield part
                if thread_id:
                    stored = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                    stored.append({"role": "assistant", "content": full_content})
                    _thread_store_put(thread_id, stored)
            except HTTPException:
                raise
            except Exception as e:
                yield f"\n[Error: {str(e)}]"

        return StreamingResponse(generate_cloud(), media_type="text/plain")

    # Prevent using the embedding-only model as a chat model.
    if request.model == RAG_EMBEDDING_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is configured as the embedding model for RAG and cannot be used for chat. Please select a chat model (e.g. '{DEFAULT_CHAT_MODEL}').",
        )
    # Only the configured default chat model may be used without an explicit Load; others must be running (ollama ps).
    if not _ollama_model_labels_equivalent(request.model, DEFAULT_CHAT_MODEL):
        if not await _ollama_chat_model_is_loaded(request.model):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{request.model}' is not loaded in Ollama. "
                    "Open Manage → Local Models and click Load for this model first."
                ),
            )

    if want_structured_output:
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="structured_output requires stream=false (Ollama structured output is non-streaming).",
            )
        try:
            try:
                rag_msgs = _messages_for_cloud_chat_with_rag(
                    messages,
                    thinking=bool(request.thinking),
                    model=request.model,
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            api_msgs = [{"role": m["role"], "content": str(m.get("content") or "")} for m in rag_msgs]
            custom_tool = (
                request.structured_tool.model_dump() if request.structured_tool is not None else None
            )
            full_content, structured = await ollama_structured_chat_complete(
                request.model,
                api_msgs,
                custom_tool=custom_tool,
                keep_alive=OLLAMA_KEEP_ALIVE,
                thinking=bool(request.thinking),
            )
            if thread_id:
                stored = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                stored.append({"role": "assistant", "content": full_content})
                _thread_store_put(thread_id, stored)
            return {
                "message": {"role": "assistant", "content": full_content},
                "structured": structured,
                "model": request.model,
            }
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if not request.stream:
        try:
            full_content = await _ollama_chat_complete_text(
                request.model, messages, thinking=bool(request.thinking)
            )
            if thread_id:
                stored = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                stored.append({"role": "assistant", "content": full_content})
                _thread_store_put(thread_id, stored)
            return {
                "message": {"role": "assistant", "content": full_content},
                "model": request.model,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def generate_response():
        full_content = ""
        timing: dict = {}
        try:
            async for part in _ollama_chat_stream_chunks(
                request.model,
                messages,
                thinking=bool(request.thinking),
                timing=timing,
            ):
                full_content += part
                yield part
            t_end = timing.get("t_end", time.perf_counter())
            t_start = timing.get("t_start", t_end)
            t_first = timing.get("t_first_public")
            wall_total = max(0.0, t_end - t_start)
            rm = timing.get("ollama_response_metadata") or {}
            eval_ns = rm.get("eval_duration")
            raw = timing.get("raw_model_output") or ""
            if _is_gemma4_model(request.model):
                if t_first is not None:
                    wall_think = max(0.0, t_first - t_start)
                    wall_answer = max(0.0, t_end - t_first)
                else:
                    wall_think = 0.0
                    wall_answer = wall_total
                ev_think, ev_answer = _gemma4_eval_think_answer_from_raw(
                    raw,
                    eval_ns if isinstance(eval_ns, int) else None,
                )
            else:
                wall_think = None
                wall_answer = wall_total
                ev_think, ev_answer = None, None
            _log_ollama_chat_finished(
                model=request.model,
                thinking_enabled=bool(request.thinking),
                streaming=True,
                wall_total_s=wall_total,
                wall_think_s=wall_think,
                wall_answer_s=wall_answer,
                ollama_eval_think_s=ev_think,
                ollama_eval_answer_s=ev_answer,
                response_metadata=rm,
            )
            if thread_id:
                stored = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                stored.append({"role": "assistant", "content": full_content})
                _thread_store_put(thread_id, stored)
        except HTTPException:
            raise
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(generate_response(), media_type="text/plain")


@app.get(
    "/api/chat/threads/{thread_id}",
    summary="Get stored messages for a thread",
)
def get_thread_messages(
    thread_id: Annotated[
        str,
        Path(description="Same value as `thread_id` sent on POST /api/chat."),
    ],
):
    """Return stored short-term memory (message history) for a thread, if any."""
    if thread_id not in _thread_store:
        return {"messages": []}
    _thread_store.move_to_end(thread_id)
    return {"messages": _thread_store[thread_id]}


# ─── Cloud LLM Streaming Helpers ───

async def stream_openai(request: CloudChatRequest):
    lc = _chat_api_messages_to_langchain(request.messages)
    re = _openai_reasoning_effort_optional(request.reasoning)
    try:
        llm = ChatOpenAI(
            model=request.model,
            api_key=request.api_key,
            timeout=httpx.Timeout(30.0, read=120.0),
            max_retries=1,
            streaming=True,
            **({"reasoning_effort": re} if re else {}),
        )
        async for chunk in llm.astream(lc):
            piece = _assistant_content_from_invoke(chunk)
            if piece:
                yield piece
    except Exception as e:
        yield f"\n[Error: OpenAI {e}]"


async def stream_anthropic(request: CloudChatRequest):
    system_text = ""
    messages = []
    for msg in request.messages:
        if msg.role == "system":
            system_text = msg.content
        else:
            messages.append({"role": msg.role, "content": msg.content})

    payload = {
        "model": request.model,
        "max_tokens": 4096,
        "messages": messages,
        "stream": True,
    }
    if system_text:
        payload["system"] = system_text

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": request.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=httpx.Timeout(30.0, read=120.0),
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"\n[Error: Anthropic API {response.status_code}: {error_body.decode()}]"
                return
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        chunk = json.loads(data)
                        if chunk.get("type") == "content_block_delta":
                            text = chunk.get("delta", {}).get("text", "")
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue


async def stream_google(request: CloudChatRequest):
    contents = []
    system_text = ""
    for msg in request.messages:
        if msg.role == "system":
            system_text = msg.content
        else:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

    payload = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 4096},
    }
    if system_text:
        payload["system_instruction"] = {"parts": [{"text": system_text}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:streamGenerateContent?alt=sse"
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": request.api_key,
            },
            json=payload,
            timeout=httpx.Timeout(30.0, read=120.0),
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"\n[Error: Google API {response.status_code}: {error_body.decode()}]"
                return
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        chunk = json.loads(data)
                        parts = chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue


CLOUD_STREAMERS = {
    "openai": stream_openai,
    "anthropic": stream_anthropic,
    "google": stream_google,
}


@app.post(
    "/api/cloud/chat",
    summary="Stream chat (client API key)",
    response_description="text/plain streamed tokens (same as streaming /api/chat).",
)
async def cloud_chat(request: CloudChatRequest):
    """Stream from OpenAI, Anthropic, or **Google** using the **client-supplied** `api_key` (e.g. browser)."""
    if request.provider not in CLOUD_STREAMERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key is required for cloud providers")
    streamer = CLOUD_STREAMERS[request.provider]
    return StreamingResponse(streamer(request), media_type="text/plain")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running!"}


def _api_usage_guide_html() -> str:
    """Static HTML overview of endpoints and parameters (complements OpenAPI schemas below)."""
    return """
  <section class="guide" id="api-usage-guide" aria-label="How to use this API">
    <h2>How to use this API</h2>
    <p class="muted">
      Use <code>Content-Type: application/json</code> for JSON bodies. Default base URL when developing is
      <code>http://localhost:8000</code>. Interactive docs: <a href="/docs">/docs</a> (Swagger),
      <a href="/redoc">/redoc</a> (ReDoc).
    </p>

    <h3>Chat</h3>
    <table class="guide">
      <thead><tr><th>Method &amp; path</th><th>Purpose</th><th>Key inputs</th></tr></thead>
      <tbody>
        <tr>
          <td><code>POST /api/chat</code></td>
          <td>Main chat: local Ollama or cloud OpenAI/Anthropic (server env keys).</td>
          <td>
            <strong>model</strong> — Ollama id or cloud model id.<br/>
            <strong>messages</strong> — <code>[{ "role", "content" }]</code>.<br/>
            <strong>stream</strong> — <code>true</code> → <code>text/plain</code> stream;
            <code>false</code> → JSON <code>message</code>, <code>model</code>, and if cloud <code>provider</code>.<br/>
            <strong>cloud</strong> — force cloud when model id is ambiguous.<br/>
            <strong>thread_id</strong> — optional; server stores last messages per id (see GET thread).<br/>
            <strong>instruction</strong> — optional; prepended to last user message for this request only.<br/>
            <strong>structured_output</strong> — optional JSON field; omit or null for false. With
            <code>stream: false</code>, uses LangChain structured output (local Ollama or cloud OpenAI/Anthropic).
            Default tool: <code>deliver_chat_response</code> → <code>structured.segments</code>.
            <strong>structured_tool</strong> — optional (local Ollama only): custom <code>name</code> +
            JSON Schema <code>parameters</code> → <code>structured.tool_name</code> +
            <code>structured.arguments</code>.<br/>
            <strong>thinking</strong> — Ollama: LangChain <code>ChatOllama</code> <code>reasoning</code> (non-Gemma) or Gemma 4
            <code>&lt;|think|&gt;</code> + strip; structured Gemma uses <code>think=false</code> when <code>false</code>.<br/>
            <strong>reasoning</strong> — OpenAI cloud: <code>none</code> | <code>low</code> | <code>medium</code> | <code>high</code> | <code>xhigh</code>
            → LangChain <code>ChatOpenAI.reasoning_effort</code> (<code>none</code> omits the parameter).
          </td>
        </tr>
        <tr>
          <td><code>POST /api/cloud/chat</code></td>
          <td>Stream chat using the <strong>client</strong> <code>api_key</code> (e.g. Google Gemini in the browser).</td>
          <td>
            <strong>provider</strong> — <code>openai</code> | <code>anthropic</code> | <code>google</code>.<br/>
            <strong>model</strong>, <strong>messages</strong>, <strong>api_key</strong>.<br/>
            <strong>reasoning</strong> — optional; OpenAI only (same as <code>/api/chat</code>).
          </td>
        </tr>
        <tr>
          <td><code>GET /api/chat/threads/{thread_id}</code></td>
          <td>Return stored messages for a thread (if any).</td>
          <td>Path: <strong>thread_id</strong> — same as sent in <code>POST /api/chat</code>.</td>
        </tr>
      </tbody>
    </table>

    <h3>RAG (knowledge base)</h3>
    <table class="guide">
      <thead><tr><th>Method &amp; path</th><th>Purpose</th><th>Key inputs</th></tr></thead>
      <tbody>
        <tr>
          <td><code>POST /api/rag/documents</code></td>
          <td>Add plain texts (chunked + embedded).</td>
          <td>Body: <code>{ "texts": ["..."] }</code></td>
        </tr>
        <tr>
          <td><code>POST /api/rag/documents/files</code></td>
          <td>Upload <code>.md</code> / <code>.txt</code> files (multipart).</td>
          <td>Form field <code>files</code> — one or more files.</td>
        </tr>
        <tr>
          <td><code>GET /api/rag/status</code></td>
          <td>Collection stats / embedding model.</td>
          <td>No parameters.</td>
        </tr>
        <tr>
          <td><code>GET /api/rag/chunks</code></td>
          <td>List stored chunks (paginated).</td>
          <td>Query: <strong>limit</strong> (1–2000, default 500), <strong>offset</strong> (default 0).</td>
        </tr>
        <tr>
          <td><code>DELETE /api/rag/chunks</code></td>
          <td>Delete chunks by id.</td>
          <td>Body: <code>{ "ids": ["..."] }</code></td>
        </tr>
        <tr>
          <td><code>DELETE /api/rag/storage</code></td>
          <td>Wipe the whole vector store.</td>
          <td>No body.</td>
        </tr>
      </tbody>
    </table>

    <h3>Ollama models &amp; system</h3>
    <table class="guide">
      <thead><tr><th>Method &amp; path</th><th>Purpose</th><th>Key inputs</th></tr></thead>
      <tbody>
        <tr><td><code>GET /api/models</code></td><td>List installed Ollama models.</td><td>—</td></tr>
        <tr><td><code>GET /api/models/loaded</code></td><td>Models currently in memory.</td><td>—</td></tr>
        <tr><td><code>POST /api/models/load</code></td><td>Load a model.</td><td>Body: <code>{ "model": "..." }</code></td></tr>
        <tr><td><code>POST /api/models/stop</code></td><td>Stop a model.</td><td>Body: <code>{ "model": "..." }</code></td></tr>
        <tr><td><code>GET /api/sysinfo</code></td><td>Host memory stats.</td><td>—</td></tr>
        <tr><td><code>GET /api/health</code></td><td>Liveness.</td><td>—</td></tr>
      </tbody>
    </table>
  </section>
"""


def _deref_openapi_schema(openapi: dict, obj: object, depth: int = 0) -> object:
    """Inline $ref for readable documentation (depth-limited; avoids runaway recursion)."""
    if depth > 14:
        return "…"
    if isinstance(obj, dict):
        ref = obj.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            name = ref.rsplit("/", 1)[-1]
            inner = (openapi.get("components") or {}).get("schemas", {}).get(name)
            if inner is not None:
                expanded = _deref_openapi_schema(openapi, inner, depth + 1)
                if isinstance(expanded, dict):
                    return {"$ref": name, **expanded}
                return {"$ref": name, "_expanded": expanded}
        return {k: _deref_openapi_schema(openapi, v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deref_openapi_schema(openapi, i, depth + 1) for i in obj]
    return obj


def _json_block(data: object) -> str:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return f"<pre><code>{html.escape(text)}</code></pre>"


def _openapi_paths_to_html(openapi: dict) -> str:
    """Build HTML sections for each path and HTTP method."""
    paths = openapi.get("paths") or {}
    parts: List[str] = []
    for path in sorted(paths.keys()):
        path_item = paths[path] or {}
        for method, op in sorted(path_item.items(), key=lambda x: x[0]):
            if method.startswith("x-") or not isinstance(op, dict):
                continue
            m = method.upper()
            summary = op.get("summary") or ""
            desc = (op.get("description") or "").strip()
            op_id = op.get("operationId") or ""
            header = f"{html.escape(m)} <code>{html.escape(path)}</code>"
            parts.append(f'<section class="op" id="{html.escape((path + "-" + m).replace("/", "-"))}">')
            parts.append(f"<h2>{header}</h2>")
            if summary:
                parts.append(f"<p class='summary'>{html.escape(summary)}</p>")
            if op_id:
                parts.append(f"<p class='muted'>operationId: <code>{html.escape(op_id)}</code></p>")
            if desc:
                parts.append(f"<div class='desc'>{html.escape(desc)}</div>")

            params = op.get("parameters") or []
            if params:
                parts.append(
                    "<h3>Parameters</h3><table><thead><tr><th>Name</th><th>In</th><th>Required</th>"
                    "<th>Description</th><th>Schema</th></tr></thead><tbody>"
                )
                for p in params:
                    if not isinstance(p, dict):
                        continue
                    name = html.escape(str(p.get("name", "")))
                    inn = html.escape(str(p.get("in", "")))
                    req = "yes" if p.get("required") else "no"
                    pdesc_raw = (p.get("description") or "").strip()
                    pdesc = html.escape(pdesc_raw) if pdesc_raw else "—"
                    sch = p.get("schema") or {}
                    parts.append(
                        "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                            name,
                            inn,
                            req,
                            pdesc,
                            html.escape(json.dumps(sch, ensure_ascii=False)),
                        )
                    )
                parts.append("</tbody></table>")

            body = op.get("requestBody")
            if isinstance(body, dict):
                parts.append("<h3>Request body</h3>")
                parts.append(f"<p>{html.escape(body.get('description') or '')}</p>")
                content = body.get("content") or {}
                for ct, cobj in sorted(content.items()):
                    parts.append(f"<p><strong>Content-Type:</strong> <code>{html.escape(ct)}</code></p>")
                    schema = cobj.get("schema") if isinstance(cobj, dict) else None
                    if schema is not None:
                        expanded = _deref_openapi_schema(openapi, schema)
                        parts.append(_json_block(expanded))

            responses = op.get("responses") or {}
            if responses:
                parts.append("<h3>Responses</h3>")
                for status in sorted(responses.keys(), key=lambda s: (len(str(s)), str(s))):
                    robj = responses[status]
                    if not isinstance(robj, dict):
                        continue
                    rdesc = robj.get("description") or ""
                    parts.append(f"<h4>{html.escape(str(status))}</h4>")
                    if rdesc:
                        parts.append(f"<p>{html.escape(rdesc)}</p>")
                    content = robj.get("content") or {}
                    for ct, cobj in sorted(content.items()):
                        parts.append(f"<p><strong>Content-Type:</strong> <code>{html.escape(ct)}</code></p>")
                        schema = cobj.get("schema") if isinstance(cobj, dict) else None
                        if schema is not None:
                            expanded = _deref_openapi_schema(openapi, schema)
                            parts.append(_json_block(expanded))

            parts.append("</section>")

    return "\n".join(parts)


def _build_api_documentation_page(openapi: dict) -> str:
    info = openapi.get("info") or {}
    title = html.escape(str(info.get("title", "API")))
    version = html.escape(str(info.get("version", "")))
    desc = (info.get("description") or "").strip()

    nav_paths = [
        '<li><a href="#api-usage-guide">Guide — parameters &amp; usage</a></li>'
    ]
    paths = openapi.get("paths") or {}
    for path in sorted(paths.keys()):
        path_item = paths[path] or {}
        for method in sorted(path_item.keys()):
            if method.startswith("x-"):
                continue
            if not isinstance(path_item[method], dict):
                continue
            m = method.upper()
            frag = (path + "-" + m).replace("/", "-")
            nav_paths.append(
                f'<li><a href="#{html.escape(frag)}">{html.escape(m)} {html.escape(path)}</a></li>'
            )

    body = _openapi_paths_to_html(openapi)
    nav = "\n".join(nav_paths)
    usage_guide = _api_usage_guide_html()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} — API reference</title>
  <style>
    :root {{
      --bg: #0d1117;
      --fg: #e6edf3;
      --muted: #8b949e;
      --border: #30363d;
      --accent: #58a6ff;
    }}
    body {{
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg);
      color: var(--fg);
      line-height: 1.5;
      margin: 0;
      padding: 0 1rem 3rem;
      max-width: 960px;
      margin-left: auto;
      margin-right: auto;
    }}
    header {{ border-bottom: 1px solid var(--border); padding: 1.25rem 0; margin-bottom: 1.5rem; }}
    h1 {{ margin: 0 0 0.35rem; font-size: 1.5rem; }}
    .muted {{ color: var(--muted); font-size: 0.9rem; }}
    .intro {{ margin: 0.75rem 0 0; color: var(--muted); }}
    .intro a {{ color: var(--accent); }}
    nav.toc {{
      background: #161b22;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-bottom: 2rem;
    }}
    nav.toc ul {{ list-style: none; padding-left: 0; margin: 0; column-count: 1; }}
    @media (min-width: 640px) {{ nav.toc ul {{ column-count: 2; }} }}
    nav.toc li {{ margin: 0.25rem 0; break-inside: avoid; }}
    nav.toc a {{ color: var(--accent); text-decoration: none; }}
    nav.toc a:hover {{ text-decoration: underline; }}
    section.op {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-bottom: 1.5rem;
      background: #161b22;
    }}
    section.op h2 {{ margin-top: 0; font-size: 1.1rem; }}
    section.op h2 code {{ color: #79c0ff; }}
    section.op h3 {{ font-size: 1rem; margin-top: 1.25rem; color: var(--fg); }}
    section.op h4 {{ font-size: 0.95rem; margin: 0.75rem 0 0.25rem; color: var(--muted); }}
    .summary {{ font-weight: 600; margin: 0.5rem 0 0; }}
    section.guide {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-bottom: 2rem;
      background: #0d1117;
    }}
    section.guide h2 {{ margin-top: 0; font-size: 1.2rem; }}
    section.guide h3 {{ font-size: 1rem; margin-top: 1.25rem; margin-bottom: 0.5rem; }}
    table.guide {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.35rem; }}
    table.guide th, table.guide td {{ border: 1px solid var(--border); padding: 0.5rem 0.65rem; text-align: left; vertical-align: top; }}
    table.guide th {{ background: #21262d; }}
    table.guide code {{ font-size: 0.88em; }}
    .desc {{ white-space: pre-wrap; color: var(--muted); font-size: 0.9rem; margin: 0.5rem 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th, td {{ border: 1px solid var(--border); padding: 0.45rem 0.6rem; text-align: left; vertical-align: top; }}
    th {{ background: #21262d; }}
    pre {{
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.75rem 1rem;
      overflow: auto;
      font-size: 0.8rem;
    }}
    code {{ font-family: ui-monospace, monospace; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="muted">OpenAPI {version} — generated from registered routes (inputs &amp; response schemas).</p>
    <p class="intro">Also available: <a href="/docs">Swagger UI (/docs)</a> · <a href="/redoc">ReDoc (/redoc)</a> ·
      <a href="/api/documentation?export=json">this page as JSON</a> (<code>/openapi.json</code>) ·
      <a href="/api/doc">/api/doc</a> (alias)</p>
  </header>
  {(f'<p class="desc">{html.escape(desc)}</p>' if desc else '')}
  {usage_guide}
  <nav class="toc" aria-label="Endpoints">
    <strong style="display:block;margin-bottom:0.5rem">Endpoints</strong>
    <ul>{nav}</ul>
  </nav>
  <main>{body}</main>
</body>
</html>"""


@app.get("/api/documentation", include_in_schema=False)
async def api_documentation(request: Request, export: Optional[str] = None):
    """Human-readable API reference (HTML) or full OpenAPI JSON (?export=json)."""
    schema = request.app.openapi()
    if export and export.strip().lower() == "json":
        return JSONResponse(schema)
    return HTMLResponse(_build_api_documentation_page(schema))


@app.get("/api/doc", include_in_schema=False)
async def api_doc_alias():
    """Same documentation as <code>/api/documentation</code> (permanent redirect)."""
    return RedirectResponse(url="/api/documentation", status_code=308)


@app.post("/api/rag/documents")
async def rag_add_documents(request: AddDocumentsRequest):
    """Add texts to the RAG knowledge base. Each string is split into chunks and embedded (Chroma + Ollama)."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")
    try:
        _, Document, _, RecursiveCharacterTextSplitter = _rag_imports()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        for t in request.texts:
            if not (t and t.strip()):
                continue
            chunks = splitter.split_text(t.strip())
            for c in chunks:
                docs.append(Document(page_content=c))
        if not docs:
            return {"status": "ok", "added": 0, "message": "No non-empty content to add."}
        vs = _get_vector_store()
        vs.add_documents(docs)
        return {"status": "ok", "added": len(docs), "message": f"Added {len(docs)} chunk(s) to the knowledge base."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/status")
async def rag_status():
    """Return RAG knowledge base status (embedding model and approximate doc count)."""
    try:
        vs = _get_vector_store()
        # Chroma collection has count; try to get it without loading all
        try:
            col = vs._collection
            n = col.count()
        except Exception:
            n = None
        return {
            "embedding_model": RAG_EMBEDDING_MODEL,
            "collection": RAG_COLLECTION,
            "persist_directory": _DB_DIR,
            "document_count": n,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/chunks")
async def rag_list_chunks(
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=2000,
            description="Maximum chunks to return (server caps at 2000).",
        ),
    ] = 500,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of chunks to skip (pagination)."),
    ] = 0,
):
    """List chunks in the vector store for browsing. Returns id, content preview, and metadata."""
    try:
        vs = _get_vector_store()
        # LangChain Chroma .get() delegates to _collection.get(); include limit/offset for pagination
        result = vs.get(
            include=["documents", "metadatas"],
            limit=min(max(1, limit), 2000),
            offset=max(0, offset),
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or [{}] * len(ids)
        if len(metadatas) < len(ids):
            metadatas = metadatas + [{}] * (len(ids) - len(metadatas))
        chunks = []
        for i, doc_id in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            # Preview: first 200 chars
            preview = (content or "")[:200]
            if len(content or "") > 200:
                preview += "..."
            chunks.append({"id": doc_id, "content": content, "preview": preview, "metadata": meta})
        return {"chunks": chunks, "total_returned": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/chunks")
async def rag_delete_chunks(body: DeleteChunksRequest):
    """Delete specific chunks by their IDs."""
    if not body.ids:
        raise HTTPException(status_code=400, detail="ids cannot be empty")
    try:
        vs = _get_vector_store()
        vs.delete(ids=body.ids)
        return {"status": "ok", "deleted": len(body.ids), "message": f"Deleted {len(body.ids)} chunk(s)."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/storage")
async def rag_delete_storage():
    """Delete entire vector storage (all chunks). Collection is reset to empty."""
    try:
        vs = _get_vector_store()
        vs.reset_collection()
        global _vector_store
        _vector_store = None
        return {"status": "ok", "message": "Vector storage cleared. All chunks have been deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


ALLOWED_EXTENSIONS = {".md", ".txt"}


def _allowed_file(filename: str) -> bool:
    if not filename:
        return False
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


@app.post("/api/rag/documents/files")
async def rag_add_files(files: List[UploadFile] = File(...)):
    """Add .md and .txt files to the knowledge base. Each file is split into chunks and embedded with source metadata."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    try:
        _, Document, _, RecursiveCharacterTextSplitter = _rag_imports()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        for f in files:
            if not _allowed_file(f.filename or ""):
                continue
            content = (await f.read()).decode("utf-8", errors="replace")
            if not (content and content.strip()):
                continue
            name = f.filename or "unknown"
            chunks = splitter.split_text(content.strip())
            for c in chunks:
                docs.append(Document(page_content=c, metadata={"source": name}))
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No valid .md or .txt content found. Upload only .md or .txt files with non-empty content.",
            )
        vs = _get_vector_store()
        vs.add_documents(docs)
        file_count = len(set(d.metadata.get("source") for d in docs))
        return {"status": "ok", "added": len(docs), "files_processed": file_count, "message": f"Added {len(docs)} chunk(s) from {file_count} file(s)."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StopModelRequest(BaseModel):
    model: str = Field(..., description="Ollama model name to unload (runs `ollama stop`).")


class LoadModelRequest(BaseModel):
    model: str = Field(
        ...,
        description="Ollama model to load and keep resident (chat or embedding model).",
    )

@app.get("/api/models")
async def get_models():
    try:
        models = _list_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/loaded")
async def get_loaded_models():
    client = ollama.AsyncClient()
    try:
        model_list = await client.ps()
        
        models = [m.model for m in model_list.models] if hasattr(model_list, 'models') else (
            [m.get('name') or m.get('model') for m in model_list.get('models', [])] if isinstance(model_list, dict) else []
        )
            
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/stop")
async def stop_model(request: StopModelRequest):
    try:
        # Stop model logic since the prompt wants exactly `ollama stop` command
        subprocess.run(["ollama", "stop", request.model], check=True)
        return {"status": "ok", "message": f"Model {request.model} stopped."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/load")
async def load_model(request: LoadModelRequest):
    client = ollama.AsyncClient()
    chat_err = None
    try:
        await client.chat(model=request.model, messages=[], keep_alive=-1)
        return {"status": "ok", "message": f"Model {request.model} loaded successfully."}
    except Exception as e:
        chat_err = e
    try:
        await client.embed(model=request.model, input=".", keep_alive=-1)
        return {"status": "ok", "message": f"Model {request.model} (embedding) loaded successfully."}
    except Exception as e2:
        if chat_err is not None:
            raise HTTPException(status_code=500, detail=str(chat_err))
        raise HTTPException(status_code=500, detail=str(e2))


@app.get("/api/ollama/verify-tools-json")
async def ollama_verify_tools_json(model: Optional[str] = None):
    """Run ChatOllama tool-calling and JSON-schema (LaTeX string) checks against a loaded Ollama model."""
    from ollama_tools_json_verify import run_ollama_tools_json_verification

    m = (model or "").strip() or DEFAULT_CHAT_MODEL
    if m == RAG_EMBEDDING_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{m}' is the RAG embedding model; pick a chat model.",
        )
    if not _ollama_model_labels_equivalent(m, DEFAULT_CHAT_MODEL):
        if not await _ollama_chat_model_is_loaded(m):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{m}' is not loaded in Ollama. "
                    "Load it in Manage → Local Models (or pass the default from settings: default_chat_model / local_models[0])."
                ),
            )
    try:
        _api_timestamp_logger.info(
            "Ollama verify-tools-json: running tool calling + JSON-schema (LaTeX) checks (model=%s).", m
        )
        t0 = time.perf_counter()
        out = await run_ollama_tools_json_verification(m)
        _api_timestamp_logger.info(
            "Ollama verify-tools-json finished: model=%s wall_s=%.3f all_ok=%s",
            m,
            time.perf_counter() - t0,
            out.get("all_ok"),
        )
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/sysinfo")
async def get_sysinfo():
    try:
        mem = psutil.virtual_memory()
        return {
            "memory": {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent,
                "used": mem.used,
                "free": mem.free
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
