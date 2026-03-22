import ollama
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import psutil
import subprocess
import httpx
import json

# Ollama chat uses LangChain ChatOllama (token-level streaming, native async).
# See: https://docs.langchain.com/oss/python/integrations/chat/ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ChromaDB persist directory under project root / database
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.join(_BASE_DIR, "..", "database", "chroma")
os.makedirs(_DB_DIR, exist_ok=True)

# RAG: embedding model for vector store (Ollama). Use one already loaded, e.g. embeddinggemma:latest
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "embeddinggemma:latest")
RAG_COLLECTION = "chatbot_kb"
RAG_TOP_K = 4

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
        _embeddings = OllamaEmbeddings(model=RAG_EMBEDDING_MODEL)
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

# Load the default model on startup to prevent slow first response
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Preloading model gemma3:1b...")
    client = ollama.AsyncClient()
    try:
        # Check if model is already pulled
        model_list = await client.list()
        models = [m.model for m in model_list.models] if hasattr(model_list, 'models') else (
            [m.get('name') or m.get('model') for m in model_list.get('models', [])] if isinstance(model_list, dict) else []
        )
        if hasattr(model_list, '__class__') and model_list.__class__.__name__ == 'ListResponse':
            # Support newest ollama SDK
            models = [m.model for m in model_list.models]
            
        if 'gemma3:1b' not in models and 'gemma3:1b' not in [m.split(':')[0] for m in models]:
            print("Model gemma3:1b not found locally. Pulling...")
            await client.pull('gemma3:1b')
            print("Model pulled successfully.")

        # keep_alive=-1 keeps the model loaded in memory indefinitely
        await client.chat(model="gemma3:1b", messages=[], keep_alive=-1)
        print("Model preloaded successfully.")
    except Exception as e:
        print(f"Error preloading model: {e}")
    yield

app = FastAPI(title="Ollama Chatbot API", lifespan=lifespan)

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "gemma3:1b"
    messages: List[ChatMessage]
    thread_id: Optional[str] = None  # optional; when set, conversation is stored per thread for short-term memory

class CloudChatRequest(BaseModel):
    provider: str  # "openai", "anthropic", "google"
    model: str
    messages: List[ChatMessage]
    api_key: str


class AddDocumentsRequest(BaseModel):
    """Add texts to the RAG knowledge base. Each string is split into chunks and embedded."""
    texts: List[str]


class DeleteChunksRequest(BaseModel):
    """Delete specific chunks from the vector store by their IDs."""
    ids: List[str]


# Short-term memory: keep last N messages so context stays focused and within model limits
# See: https://docs.langchain.com/oss/python/langchain/short-term-memory
MAX_CONTEXT_MESSAGES = int(os.environ.get("MAX_CONTEXT_MESSAGES", "30"))
_thread_store: dict[str, list] = {}  # thread_id -> list of {role, content}


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


@app.post("/api/chat")
async def chat_completion(request: ChatRequest):
    # Prevent using the embedding-only model as a chat model.
    if request.model == RAG_EMBEDDING_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is configured as the embedding model for RAG and cannot be used for chat. Please select a chat model (e.g. 'gemma3:1b').",
        )
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    messages = _trim_messages(messages)
    thread_id = request.thread_id

    async def generate_response():
        full_content = ""
        try:
            # 2-Step RAG: always retrieve for the latest user message, then generate with context.
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
                raise  # e.g. 503 RAG deps not available
            except Exception:
                pass  # no RAG context on embedding/Chroma errors
            # Always inject RAG system message: retrieved context is the source of truth
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
            lc_messages = [SystemMessage(content=rag_system)] + lc_messages
            llm = ChatOllama(
                model=request.model,
                temperature=0,
            )
            async for chunk in llm.astream(lc_messages):
                if chunk.content:
                    full_content += chunk.content
                    yield chunk.content
            # Persist short-term memory for this thread when streaming finishes
            if thread_id:
                stored = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                stored.append({"role": "assistant", "content": full_content})
                _thread_store[thread_id] = stored
        except HTTPException:
            raise
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(generate_response(), media_type="text/plain")


@app.get("/api/chat/threads/{thread_id}")
def get_thread_messages(thread_id: str):
    """Return stored short-term memory (message history) for a thread, if any."""
    if thread_id not in _thread_store:
        return {"messages": []}
    return {"messages": _thread_store[thread_id]}


# ─── Cloud LLM Streaming Helpers ───

async def stream_openai(request: CloudChatRequest):
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {request.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": request.model, "messages": messages, "stream": True},
            timeout=httpx.Timeout(30.0, read=120.0),
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"\n[Error: OpenAI API {response.status_code}: {error_body.decode()}]"
                return
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


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


@app.post("/api/cloud/chat")
async def cloud_chat(request: CloudChatRequest):
    if request.provider not in CLOUD_STREAMERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key is required for cloud providers")
    streamer = CLOUD_STREAMERS[request.provider]
    return StreamingResponse(streamer(request), media_type="text/plain")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running!"}


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
async def rag_list_chunks(limit: int = 500, offset: int = 0):
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
    model: str

class LoadModelRequest(BaseModel):
    model: str

@app.get("/api/models")
async def get_models():
    client = ollama.AsyncClient()
    try:
        model_list = await client.list()
        models = [m.model for m in model_list.models] if hasattr(model_list, 'models') else (
            [m.get('name') or m.get('model') for m in model_list.get('models', [])] if isinstance(model_list, dict) else []
        )
        if hasattr(model_list, '__class__') and model_list.__class__.__name__ == 'ListResponse':
            models = [m.model for m in model_list.models]
            
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
