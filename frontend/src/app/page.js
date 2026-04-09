"use client";

import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/**
 * LLMs often emit LaTeX without proper $ / $$ delimiters. Normalize before remark-math.
 */
function preprocessMathForMarkdown(text) {
  if (!text || typeof text !== 'string') return text;
  let t = text;
  // LaTeX display \[ ... \]
  t = t.replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => `\n\n$$\n${inner.trim()}\n$$\n\n`);
  // LaTeX inline \( ... \)
  t = t.replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner.trim()}$`);
  // Display block: [ ... ] on its own lines with TeX inside (e.g. [\n\sum_{...}\n])
  // Display block: newline, [, newline, TeX lines, newline, ] (common LLM output)
  t = t.replace(/\n\[\s*\n([\s\S]*?)\n\]/g, (match, inner) => {
    const trimmed = inner.trim();
    if (/\\[a-zA-Z]/.test(trimmed)) {
      return `\n\n$$\n${trimmed}\n$$\n\n`;
    }
    return match;
  });
  // Same when [ starts the message
  t = t.replace(/^\[\s*\n([\s\S]*?)\n\]/m, (match, inner) => {
    const trimmed = inner.trim();
    if (/\\[a-zA-Z]/.test(trimmed)) {
      return `$$\n${trimmed}\n$$\n\n`;
    }
    return match;
  });
  return t;
}

const CLOUD_MODELS = {
  openai: [
    'gpt-5.4',
    'gpt-5.4-pro',
    'gpt-5-mini',
    'gpt-5-nano',
    'gpt-5',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-4o',
    'gpt-4o-mini',
    'o4-mini',
    'o3-mini',
    'o3',
  ],
  anthropic: [
    'claude-opus-4-6',
    'claude-sonnet-4-6',
    'claude-haiku-4-5-20251001',
    'claude-sonnet-4-5-20250929',
    'claude-opus-4-5-20251101',
    'claude-opus-4-1-20250805',
    'claude-sonnet-4-20250514',
    'claude-opus-4-20250514',
  ],
  google: [
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemini-2.0-flash',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
  ],
};

const PROVIDER_LABELS = {
  ollama: 'Ollama',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  google: 'Google',
};

const LOCAL_OLLAMA_MODEL_OPTIONS = [
  'gpt-oss:20b',
  'qwen3.5:27b',
];

function mergeUniqueModelNames(...groups) {
  const out = [];
  for (const group of groups) {
    for (const name of group || []) {
      if (!name || out.includes(name)) continue;
      out.push(name);
    }
  }
  return out;
}

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [models, setModels] = useState([]);
  const [loadedModels, setLoadedModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('gpt-oss:20b');
  const [manageModel, setManageModel] = useState('gpt-oss:20b');
  const [isManageModalOpen, setIsManageModalOpen] = useState(false);

  // Cloud provider state
  const [provider, setProvider] = useState('ollama');
  const [apiKeys, setApiKeys] = useState({ openai: '', anthropic: '', google: '' });
  const [cloudModels, setCloudModels] = useState({
    openai: 'gpt-5.4',
    anthropic: 'claude-sonnet-4-6',
    google: 'gemini-2.5-flash',
  });
  const [showApiKey, setShowApiKey] = useState({ openai: false, anthropic: false, google: false });
  const [settingsTab, setSettingsTab] = useState('local'); // 'local' | 'cloud' | 'rag' | 'embeddings'
  const [ragStatus, setRagStatus] = useState(null);
  const [ragError, setRagError] = useState(null);
  const [ragText, setRagText] = useState('');
  const [ragAdding, setRagAdding] = useState(false);

  // Embeddings / vector storage tab
  const [embeddingsChunks, setEmbeddingsChunks] = useState([]);
  const [embeddingsChunkCount, setEmbeddingsChunkCount] = useState(null);
  const [selectedChunkIds, setSelectedChunkIds] = useState([]);
  const [confirmClearStorage, setConfirmClearStorage] = useState(false);
  const [embeddingsError, setEmbeddingsError] = useState(null);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [embeddingsActionLoading, setEmbeddingsActionLoading] = useState(false);
  const [fileUploadProgress, setFileUploadProgress] = useState(null);
  const fileInputRef = useRef(null);
  const directoryInputRef = useRef(null);

  const [threadId, setThreadId] = useState(() =>
    typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `thread-${Date.now()}`
  );
  const [autoScroll, setAutoScroll] = useState(true);
  const endOfMessagesRef = useRef(null);
  const chatHistoryRef = useRef(null);
  const lastScrollTopRef = useRef(0);
  const currentRequestControllerRef = useRef(null);
  const localModelOptions = mergeUniqueModelNames(LOCAL_OLLAMA_MODEL_OPTIONS, models);

  // Load API keys from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('chatbot-api-keys');
      if (saved) setApiKeys(JSON.parse(saved));
      const savedProvider = localStorage.getItem('chatbot-provider');
      if (savedProvider) setProvider(savedProvider);
      const savedCloudModels = localStorage.getItem('chatbot-cloud-models');
      if (savedCloudModels) {
        const parsed = JSON.parse(savedCloudModels);
        const normalized = { ...parsed };
        (['openai', 'anthropic', 'google']).forEach(prov => {
          if (!CLOUD_MODELS[prov].includes(parsed[prov])) {
            normalized[prov] = CLOUD_MODELS[prov][0];
          }
        });
        setCloudModels(normalized);
      }
    } catch {}
  }, []);

  // Persist provider choice
  useEffect(() => {
    localStorage.setItem('chatbot-provider', provider);
  }, [provider]);

  // Persist cloud model choices
  useEffect(() => {
    localStorage.setItem('chatbot-cloud-models', JSON.stringify(cloudModels));
  }, [cloudModels]);

  const saveApiKey = (prov, key) => {
    const updated = { ...apiKeys, [prov]: key };
    setApiKeys(updated);
    localStorage.setItem('chatbot-api-keys', JSON.stringify(updated));
  };

  useEffect(() => {
    if (isManageModalOpen && (settingsTab === 'rag' || settingsTab === 'embeddings')) {
      setRagError(null);
      setEmbeddingsError(null);
      fetch('http://localhost:8000/api/rag/status')
        .then(r => {
          if (!r.ok) return r.json().then(data => { setRagError(data.detail || r.statusText); setRagStatus(null); setEmbeddingsChunkCount(null); });
          return r.json().then(data => { setRagStatus(data); setRagError(null); setEmbeddingsChunkCount(data.document_count); });
        })
        .catch(() => { setRagStatus(null); setEmbeddingsChunkCount(null); setRagError('Could not reach backend. Start it with: cd backend && ./run.sh'); });
    }
  }, [isManageModalOpen, settingsTab]);

  useEffect(() => {
    if (isManageModalOpen && settingsTab === 'embeddings') {
      setChunksLoading(true);
      setEmbeddingsError(null);
      fetch('http://localhost:8000/api/rag/chunks?limit=500&offset=0')
        .then(r => {
          if (!r.ok) return r.json().then(data => { setEmbeddingsError(data.detail || r.statusText); setEmbeddingsChunks([]); });
          return r.json().then(data => { setEmbeddingsChunks(data.chunks || []); setEmbeddingsError(null); });
        })
        .catch(() => { setEmbeddingsChunks([]); setEmbeddingsError('Failed to load chunks'); })
        .finally(() => setChunksLoading(false));
    }
  }, [isManageModalOpen, settingsTab]);

  const refreshOllamaModelState = useCallback(() => {
    fetch('http://localhost:8000/api/models')
      .then(r => r.json())
      .then(data => {
        if (Array.isArray(data.models)) {
          const merged = mergeUniqueModelNames(LOCAL_OLLAMA_MODEL_OPTIONS, data.models);
          setModels(merged);
          setManageModel(prev => prev && merged.includes(prev) ? prev : merged[0]);
        }
      })
      .catch(err => console.error("Error fetching models:", err));

    fetch('http://localhost:8000/api/models/loaded')
      .then(r => r.json())
      .then(data => {
        if (data.models) {
          setLoadedModels(data.models);
          setSelectedModel(prev => {
            if (data.models.includes(prev)) return prev;
            return data.models.length > 0 ? data.models[0] : (prev || 'gpt-oss:20b');
          });
        }
      })
      .catch(err => console.error("Error fetching loaded models:", err));
  }, []);

  useEffect(() => {
    refreshOllamaModelState();
  }, [refreshOllamaModelState]);

  useEffect(() => {
    if (isManageModalOpen && settingsTab === 'local') {
      refreshOllamaModelState();
    }
  }, [isManageModalOpen, settingsTab, refreshOllamaModelState]);

  const handleStopModel = async () => {
    if (!manageModel) return;
    try {
      const response = await fetch('http://localhost:8000/api/models/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: manageModel })
      });
      if (!response.ok) throw new Error("Failed to stop model");
      alert(`Model ${manageModel} stopped successfully`);
      refreshOllamaModelState();
    } catch (err) {
      alert(err.message);
    }
  };

  const handleLoadModel = async () => {
    if (!manageModel) return;
    try {
      const response = await fetch('http://localhost:8000/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: manageModel })
      });
      if (!response.ok) throw new Error("Failed to load model");
      alert(`Model ${manageModel} loaded successfully`);
      refreshOllamaModelState();
    } catch (err) {
      alert(err.message);
    }
  };

  // Auto-scroll to bottom only when user is near the bottom
  useEffect(() => {
    if (autoScroll) {
      endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading, autoScroll]);

  const currentModelName = provider === 'ollama' ? selectedModel : cloudModels[provider];

  const handleStopStreaming = () => {
    if (currentRequestControllerRef.current) {
      currentRequestControllerRef.current.abort();
      currentRequestControllerRef.current = null;
    }
    setIsLoading(false);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (isLoading) {
      handleStopStreaming();
      return;
    }
    if (!input.trim()) return;

    // Google still uses the browser-stored key on /api/cloud/chat; OpenAI/Anthropic use server .env.local via /api/chat.
    if (provider === 'google' && !apiKeys[provider]) {
      setError(`Please set your ${PROVIDER_LABELS.google} API key in Settings → Cloud API Keys.`);
      return;
    }

    const userMsg = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMsg];

    setMessages(newMessages);
    setInput('');
    setIsLoading(true);
    setError(null);

    const controller = new AbortController();
    currentRequestControllerRef.current = controller;

    try {
      let response;

      if (provider === 'ollama') {
        response = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            model: selectedModel,
            messages: newMessages,
            thread_id: threadId,
            stream: true,
          }),
        });
      } else if (provider === 'openai' || provider === 'anthropic') {
        response = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            cloud: true,
            model: cloudModels[provider],
            messages: newMessages,
            thread_id: threadId,
            stream: true,
          }),
        });
      } else {
        response = await fetch('http://localhost:8000/api/cloud/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            provider,
            model: cloudModels[provider],
            messages: newMessages,
            api_key: apiKeys[provider],
          }),
        });
      }

      if (!response.ok) {
        let message = `API error: ${response.status} ${response.statusText}`;
        try {
          const data = await response.json();
          if (data && data.detail) {
            message = data.detail;
          }
        } catch {
          // ignore JSON parse errors and keep default message
        }
        setError(message);
        return;
      }

      const contentType = response.headers.get('content-type') || '';

      if (contentType.includes('application/json')) {
        const data = await response.json();
        const content =
          typeof data?.message?.content === 'string' ? data.message.content : '';
        setMessages((prev) => [...prev, { role: 'assistant', content }]);
      } else {
        // Default: incremental plain-text stream (Ollama, cloud via /api/chat, Google /api/cloud/chat)
        setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          const textChunk = decoder.decode(value, { stream: true });

          setMessages((prev) => {
            const allExceptLast = prev.slice(0, prev.length - 1);
            const lastMsg = prev[prev.length - 1];
            return [
              ...allExceptLast,
              { ...lastMsg, content: lastMsg.content + textChunk },
            ];
          });
        }
      }
    } catch (err) {
      if (err?.name === 'AbortError') {
        return;
      }
      console.error(err);
      setError(
        provider === 'ollama'
          ? "Failed to communicate with the backend. Ensure it is running."
          : provider === 'google'
            ? `Failed to get response from ${PROVIDER_LABELS[provider]}. Check your API key and model.`
            : `Failed to get response from ${PROVIDER_LABELS[provider]}. Ensure OPENAI_API_KEY or ANTHROPIC_API_KEY is set in backend .env.local and the model id is valid.`
      );
    } finally {
      if (currentRequestControllerRef.current === controller) {
        currentRequestControllerRef.current = null;
      }
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  };

  return (
    <>
      <main className="container">
        {/* Header */}
        <header className="header" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', position: 'relative' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="header-title" style={{ flex: 1, textAlign: 'center' }}>
              <span>✨</span> Ollama Chat
            </div>

            <button
              onClick={() => setIsManageModalOpen(true)}
              style={{ position: 'absolute', top: '10px', right: '15px', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)' }}
              title="Settings"
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
            </button>
          </div>

          {/* Provider Tabs */}
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: '0.2rem' }}>
            <div className="provider-tabs">
              {Object.entries(PROVIDER_LABELS).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => {
                    setProvider(key);
                    if (key === 'ollama') refreshOllamaModelState();
                  }}
                  className={`provider-tab ${provider === key ? 'active' : ''}`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Model Selector */}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-secondary)', padding: '0.3rem 0.6rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
              <button
                type="button"
                onClick={() => { setMessages([]); setThreadId(typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `thread-${Date.now()}`); setError(null); }}
                className="model-badge"
                style={{ cursor: 'pointer', background: 'transparent', border: 'none', outline: 'none', padding: '0 0.25rem', color: 'var(--text-secondary)', fontSize: '0.85rem' }}
                title="Start a new conversation (new thread)"
              >
                New chat
              </button>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Model:</span>
              {provider === 'ollama' ? (
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                  className="model-badge"
                  style={{ cursor: 'pointer', background: 'transparent', border: 'none', outline: 'none', padding: 0 }}
                >
                  {loadedModels.length === 0 && <option value="none">No models loaded</option>}
                  {loadedModels.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              ) : (
                <select
                  value={cloudModels[provider]}
                  onChange={e => setCloudModels(prev => ({ ...prev, [provider]: e.target.value }))}
                  className="model-badge"
                  style={{ cursor: 'pointer', background: 'transparent', border: 'none', outline: 'none', padding: 0 }}
                >
                  {CLOUD_MODELS[provider].map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              )}
              {/* Google requires a browser API key; OpenAI/Anthropic use server .env.local */}
              {provider === 'google' && !apiKeys[provider] && (
                <span title="API key not set" style={{ color: '#ef4444', fontSize: '0.7rem', cursor: 'help' }}>⚠</span>
              )}
            </div>
          </div>
        </header>

        {/* Chat Messages */}
        <div
          className="chat-history"
          ref={chatHistoryRef}
          onScroll={() => {
            const el = chatHistoryRef.current;
            if (!el) return;
            const currentTop = el.scrollTop;
            const isAtBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
            const isScrollingUp = currentTop < lastScrollTopRef.current;
            if (isAtBottom) {
              setAutoScroll(true);
            } else if (isScrollingUp) {
              setAutoScroll(false);
            }
            lastScrollTopRef.current = currentTop;
          }}
        >
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: 'auto', marginBottom: 'auto' }}>
              <h2>Welcome to Ollama Chatbot!</h2>
              <p style={{ marginTop: '0.5rem', opacity: 0.8 }}>
                {provider === 'ollama'
                  ? 'Type a message below to start chatting.'
                  : `Chatting with ${PROVIDER_LABELS[provider]} (${cloudModels[provider]}). Type a message below.`}
              </p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message-wrapper ${msg.role === 'user' ? 'user' : 'bot'}`}>
              <div className="message-bubble">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
                  components={{
                    p: ({ children }) => (
                      <p style={{ margin: '0.3rem 0' }}>
                        {children}
                      </p>
                    ),
                    ul: ({ children }) => (
                      <ul style={{ paddingLeft: '1.25rem', margin: '0.25rem 0' }}>
                        {children}
                      </ul>
                    ),
                    ol: ({ children }) => (
                      <ol style={{ paddingLeft: '1.25rem', margin: '0.25rem 0' }}>
                        {children}
                      </ol>
                    ),
                    li: ({ children }) => (
                      <li style={{ marginBottom: '0.15rem' }}>
                        {children}
                      </li>
                    ),
                    strong: ({ children }) => (
                      <strong style={{ fontWeight: 600 }}>
                        {children}
                      </strong>
                    ),
                  }}
                >
                  {preprocessMathForMarkdown(msg.content)}
                </ReactMarkdown>
              </div>
            </div>
          ))}

          {isLoading && messages.length > 0 && messages[messages.length - 1].role !== 'assistant' && (
            <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠ {error}
            </div>
          )}

          <div ref={endOfMessagesRef} />
        </div>

        {/* Input Area */}
        <div className="input-area">
          <form onSubmit={handleSend} className="input-container">
            <textarea
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              rows="1"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!isLoading && !input.trim()}
              aria-label={isLoading ? "Stop response" : "Send Message"}
            >
              {isLoading ? (
                <svg className="send-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
                </svg>
              ) : (
                <svg className="send-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </button>
          </form>
        </div>
      </main>

      {/* Settings Modal */}
      {isManageModalOpen && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(4px)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 9999
        }}>
          <div style={{
            backgroundColor: '#161b22', padding: '2rem', borderRadius: '12px', minWidth: '420px', maxWidth: '500px',
            border: '1px solid var(--chat-border)', boxShadow: '0 10px 25px rgba(0,0,0,0.8)', color: 'var(--text-primary)',
            maxHeight: '80vh', overflowY: 'auto'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Settings</h2>
              <button
                onClick={() => setIsManageModalOpen(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '1.2rem' }}
              >
                ✕
              </button>
            </div>

            {/* Settings Tabs */}
            <div className="settings-tabs" style={{ marginBottom: '1.5rem' }}>
              <button
                onClick={() => setSettingsTab('local')}
                className={`settings-tab ${settingsTab === 'local' ? 'active' : ''}`}
              >
                Local Models
              </button>
              <button
                onClick={() => setSettingsTab('cloud')}
                className={`settings-tab ${settingsTab === 'cloud' ? 'active' : ''}`}
              >
                Cloud API Keys
              </button>
              <button
                onClick={() => setSettingsTab('rag')}
                className={`settings-tab ${settingsTab === 'rag' ? 'active' : ''}`}
              >
                Knowledge base
              </button>
              <button
                onClick={() => setSettingsTab('embeddings')}
                className={`settings-tab ${settingsTab === 'embeddings' ? 'active' : ''}`}
              >
                Embeddings & storage
              </button>
            </div>

            {/* Local Models Tab */}
            {settingsTab === 'local' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Select Model</label>
                  <select
                    value={manageModel}
                    onChange={e => setManageModel(e.target.value)}
                    className="chat-input"
                    style={{ cursor: 'pointer', padding: '0.5rem' }}
                  >
                    {localModelOptions.length === 0 && <option value="none">No models available</option>}
                    {localModelOptions.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>

                <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
                  <button
                    onClick={handleLoadModel}
                    disabled={loadedModels.includes(manageModel)}
                    style={{
                      flex: 1, padding: '0.5rem', borderRadius: '6px', fontWeight: 'bold',
                      background: loadedModels.includes(manageModel) ? 'var(--bg-secondary)' : '#10b981',
                      color: loadedModels.includes(manageModel) ? 'var(--text-secondary)' : 'white',
                      border: '1px solid var(--border)',
                      cursor: loadedModels.includes(manageModel) ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Load Model
                  </button>
                  <button
                    onClick={handleStopModel}
                    disabled={!loadedModels.includes(manageModel)}
                    style={{
                      flex: 1, padding: '0.5rem', borderRadius: '6px', fontWeight: 'bold',
                      background: !loadedModels.includes(manageModel) ? 'var(--bg-secondary)' : '#ef4444',
                      color: !loadedModels.includes(manageModel) ? 'var(--text-secondary)' : 'white',
                      border: !loadedModels.includes(manageModel) ? '1px solid var(--border)' : 'none',
                      cursor: !loadedModels.includes(manageModel) ? 'not-allowed' : 'pointer'
                    }}
                  >
                    Stop Model
                  </button>
                </div>

                <div style={{ marginTop: '0.25rem', fontSize: '0.85rem', textAlign: 'center', color: loadedModels.includes(manageModel) ? '#10b981' : 'var(--text-secondary)' }}>
                  Status: {loadedModels.includes(manageModel) ? 'Loaded in Memory' : 'Offline'}
                </div>
              </div>
            )}

            {/* Cloud API Keys Tab */}
            {settingsTab === 'cloud' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <p className="settings-helper-text">
                  In-app OpenAI and Anthropic chat uses <code style={{ fontSize: '0.8em' }}>OPENAI_API_KEY</code> and{' '}
                  <code style={{ fontSize: '0.8em' }}>ANTHROPIC_API_KEY</code> in the backend{' '}
                  <code style={{ fontSize: '0.8em' }}>.env.local</code>. Keys below are for Google (required) and optional use with{' '}
                  <code style={{ fontSize: '0.8em' }}>/api/cloud/chat</code>.
                </p>

                {['openai', 'anthropic', 'google'].map(prov => (
                  <div key={prov} className="api-key-section">
                    <label style={{ fontSize: '0.9rem', fontWeight: '500', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {PROVIDER_LABELS[prov]}
                      {apiKeys[prov] && (
                        <span style={{ fontSize: '0.7rem', color: '#10b981', fontWeight: '400' }}>Saved locally</span>
                      )}
                    </label>
                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.4rem' }}>
                      <input
                        type={showApiKey[prov] ? 'text' : 'password'}
                        value={apiKeys[prov]}
                        onChange={e => saveApiKey(prov, e.target.value)}
                        placeholder={
                          prov === 'openai' ? 'sk-... (main chat: backend .env.local)' :
                          prov === 'anthropic' ? 'sk-ant-... (main chat: backend .env.local)' :
                          'AI...'
                        }
                        className="api-key-input"
                      />
                      <button
                        onClick={() => setShowApiKey(prev => ({ ...prev, [prov]: !prev[prov] }))}
                        style={{
                          background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                          borderRadius: '6px', padding: '0.4rem 0.6rem', cursor: 'pointer',
                          color: 'var(--text-secondary)', fontSize: '0.8rem', whiteSpace: 'nowrap'
                        }}
                        title={showApiKey[prov] ? 'Hide' : 'Show'}
                      >
                        {showApiKey[prov] ? '🙈' : '👁'}
                      </button>
                      {apiKeys[prov] && (
                        <button
                          onClick={() => saveApiKey(prov, '')}
                          style={{
                            background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)',
                            borderRadius: '6px', padding: '0.4rem 0.6rem', cursor: 'pointer',
                            color: '#ef4444', fontSize: '0.8rem', whiteSpace: 'nowrap'
                          }}
                          title="Remove key"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Knowledge base (RAG) Tab */}
            {settingsTab === 'rag' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <p className="settings-helper-text">
                  Add text to the knowledge base. When you chat with Ollama, the app retrieves relevant chunks and uses them as context (2-step RAG). Use an embedding model already loaded in Ollama (e.g. embeddinggemma:latest) — see Local Models.
                </p>
                {ragError && (
                  <div className="api-key-section" style={{ borderColor: 'rgba(239, 68, 68, 0.5)', background: 'rgba(239, 68, 68, 0.08)' }}>
                    <span style={{ fontSize: '0.85rem', color: '#fca5a5' }}>{ragError}</span>
                  </div>
                )}
                {ragStatus && (
                  <div className="api-key-section">
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Embedding: {ragStatus.embedding_model} · Chunks: {ragStatus.document_count ?? '—'}
                    </span>
                  </div>
                )}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Add document text</label>
                  <textarea
                    className="api-key-input"
                    value={ragText}
                    onChange={e => setRagText(e.target.value)}
                    placeholder="Paste text or a paragraph to add to the knowledge base..."
                    rows={4}
                    style={{ resize: 'vertical', minHeight: '80px' }}
                  />
                  <button
                    onClick={async () => {
                      if (!ragText.trim()) return;
                      setRagAdding(true);
                      try {
                        const r = await fetch('http://localhost:8000/api/rag/documents', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ texts: [ragText.trim()] }),
                        });
                        const data = await r.json();
                        if (!r.ok) throw new Error(data.detail || 'Failed to add');
                        setRagText('');
                        if (ragStatus != null) setRagStatus(prev => prev ? { ...prev, document_count: (prev.document_count ?? 0) + data.added } : null);
                        else fetch('http://localhost:8000/api/rag/status').then(res => res.json()).then(setRagStatus);
                      } catch (err) {
                        alert(err.message || 'Failed to add to knowledge base');
                      } finally {
                        setRagAdding(false);
                      }
                    }}
                    disabled={!ragText.trim() || ragAdding}
                    style={{
                      padding: '0.5rem 1rem', borderRadius: '8px', fontWeight: '600',
                      background: ragText.trim() && !ragAdding ? 'var(--accent-color)' : 'var(--bg-secondary)',
                      color: ragText.trim() && !ragAdding ? 'white' : 'var(--text-secondary)',
                      border: '1px solid var(--border)', cursor: ragText.trim() && !ragAdding ? 'pointer' : 'not-allowed',
                    }}
                  >
                    {ragAdding ? 'Adding…' : 'Add to knowledge base'}
                  </button>
                </div>
              </div>
            )}

            {/* Embeddings & vector storage Tab */}
            {settingsTab === 'embeddings' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <p className="settings-helper-text">
                  Manage vector storage and embeddings: clear all data, browse/delete chunks, or add .md / .txt files (single, multiple, or entire folder).
                </p>
                {embeddingsError && (
                  <div className="api-key-section" style={{ borderColor: 'rgba(239, 68, 68, 0.5)', background: 'rgba(239, 68, 68, 0.08)' }}>
                    <span style={{ fontSize: '0.85rem', color: '#fca5a5' }}>{embeddingsError}</span>
                  </div>
                )}
                {ragStatus && (
                  <div className="api-key-section">
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Embedding: {ragStatus.embedding_model} · Total chunks: {embeddingsChunkCount ?? ragStatus.document_count ?? '—'}
                    </span>
                  </div>
                )}

                {/* Delete entire vector storage */}
                <div className="api-key-section">
                  <label style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-primary)' }}>Delete entire vector storage</label>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.4rem 0 0.6rem 0' }}>
                    Permanently remove all chunks. This cannot be undone.
                  </p>
                  {!confirmClearStorage ? (
                    <button
                      type="button"
                      onClick={() => setConfirmClearStorage(true)}
                      disabled={embeddingsActionLoading}
                      style={{
                        padding: '0.5rem 1rem', borderRadius: '8px', fontWeight: '600',
                        background: 'rgba(239, 68, 68, 0.2)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.4)',
                        cursor: embeddingsActionLoading ? 'not-allowed' : 'pointer',
                      }}
                    >
                      Clear all vector storage
                    </button>
                  ) : (
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Are you sure?</span>
                      <button
                        type="button"
                        onClick={async () => {
                          setEmbeddingsActionLoading(true);
                          try {
                            const r = await fetch('http://localhost:8000/api/rag/storage', { method: 'DELETE' });
                            const data = await r.json();
                            if (!r.ok) throw new Error(data.detail || 'Failed to clear');
                            setConfirmClearStorage(false);
                            setEmbeddingsChunks([]);
                            setSelectedChunkIds([]);
                            setEmbeddingsChunkCount(0);
                            if (ragStatus) setRagStatus(prev => prev ? { ...prev, document_count: 0 } : null);
                          } catch (err) {
                            alert(err.message || 'Failed to clear storage');
                          } finally {
                            setEmbeddingsActionLoading(false);
                          }
                        }}
                        style={{
                          padding: '0.4rem 0.8rem', borderRadius: '6px', fontWeight: '600',
                          background: '#dc2626', color: 'white', border: 'none', cursor: 'pointer',
                        }}
                      >
                        Yes, delete all
                      </button>
                      <button
                        type="button"
                        onClick={() => setConfirmClearStorage(false)}
                        style={{
                          padding: '0.4rem 0.8rem', borderRadius: '6px', background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border)', cursor: 'pointer',
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </div>

                {/* Browse and delete chunks */}
                <div className="api-key-section">
                  <label style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-primary)' }}>Browse chunks</label>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.4rem 0 0.6rem 0' }}>
                    Select chunks to delete. Showing up to 500.
                  </p>
                  {chunksLoading ? (
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Loading chunks…</div>
                  ) : embeddingsChunks.length === 0 ? (
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>No chunks in storage.</div>
                  ) : (
                    <>
                      <div style={{ maxHeight: '220px', overflowY: 'auto', border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--bg-secondary)' }}>
                        {embeddingsChunks.map((chunk) => (
                          <div
                            key={chunk.id}
                            style={{
                              display: 'flex',
                              alignItems: 'flex-start',
                              gap: '0.5rem',
                              padding: '0.5rem 0.6rem',
                              borderBottom: '1px solid var(--border)',
                              background: selectedChunkIds.includes(chunk.id) ? 'rgba(59, 130, 246, 0.12)' : 'transparent',
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={selectedChunkIds.includes(chunk.id)}
                              onChange={(e) => {
                                if (e.target.checked) setSelectedChunkIds(prev => [...prev, chunk.id]);
                                else setSelectedChunkIds(prev => prev.filter(id => id !== chunk.id));
                              }}
                              style={{ marginTop: '0.3rem' }}
                            />
                            <div style={{ flex: 1, minWidth: 0 }}>
                              {chunk.metadata?.source && (
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>{chunk.metadata.source}</div>
                              )}
                              <div style={{ fontSize: '0.85rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }} title={chunk.content}>
                                {chunk.preview}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <button
                        type="button"
                        disabled={selectedChunkIds.length === 0 || embeddingsActionLoading}
                        onClick={async () => {
                          if (selectedChunkIds.length === 0) return;
                          setEmbeddingsActionLoading(true);
                          try {
                            const r = await fetch('http://localhost:8000/api/rag/chunks', {
                              method: 'DELETE',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ ids: selectedChunkIds }),
                            });
                            const data = await r.json();
                            if (!r.ok) throw new Error(data.detail || 'Failed to delete');
                            setEmbeddingsChunks(prev => prev.filter(c => !selectedChunkIds.includes(c.id)));
                            setSelectedChunkIds([]);
                            setEmbeddingsChunkCount(prev => (prev != null ? Math.max(0, prev - selectedChunkIds.length) : null));
                            if (ragStatus) setRagStatus(prev => prev ? { ...prev, document_count: Math.max(0, (prev.document_count ?? 0) - selectedChunkIds.length) } : null);
                          } catch (err) {
                            alert(err.message || 'Failed to delete chunks');
                          } finally {
                            setEmbeddingsActionLoading(false);
                          }
                        }}
                        style={{
                          marginTop: '0.5rem', padding: '0.4rem 0.8rem', borderRadius: '6px', fontWeight: '600',
                          background: selectedChunkIds.length && !embeddingsActionLoading ? 'rgba(239, 68, 68, 0.2)' : 'var(--bg-secondary)',
                          color: selectedChunkIds.length && !embeddingsActionLoading ? '#fca5a5' : 'var(--text-secondary)',
                          border: '1px solid var(--border)', cursor: selectedChunkIds.length && !embeddingsActionLoading ? 'pointer' : 'not-allowed',
                        }}
                      >
                        Delete selected ({selectedChunkIds.length})
                      </button>
                    </>
                  )}
                </div>

                {/* Add from files or directory */}
                <div className="api-key-section">
                  <label style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-primary)' }}>Add .md / .txt files</label>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.4rem 0 0.6rem 0' }}>
                    Upload one or more files, or select a folder to add all .md and .txt files inside it.
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".md,.txt"
                    multiple
                    style={{ display: 'none' }}
                    onChange={async (e) => {
                      const files = e.target.files;
                      if (!files?.length) return;
                      setFileUploadProgress('Uploading…');
                      setEmbeddingsActionLoading(true);
                      try {
                        const form = new FormData();
                        for (let i = 0; i < files.length; i++) form.append('files', files[i]);
                        const r = await fetch('http://localhost:8000/api/rag/documents/files', { method: 'POST', body: form });
                        const data = await r.json();
                        if (!r.ok) throw new Error(data.detail || 'Upload failed');
                        setFileUploadProgress(`Added ${data.added} chunk(s) from ${data.files_processed} file(s).`);
                        setEmbeddingsChunkCount(prev => (prev != null ? prev + data.added : data.added));
                        if (ragStatus) setRagStatus(prev => prev ? { ...prev, document_count: (prev.document_count ?? 0) + data.added } : null);
                        fetch('http://localhost:8000/api/rag/chunks?limit=500&offset=0').then(res => res.json()).then(d => setEmbeddingsChunks(d.chunks || []));
                      } catch (err) {
                        alert(err.message || 'Upload failed');
                      } finally {
                        setEmbeddingsActionLoading(false);
                        e.target.value = '';
                        if (fileInputRef.current) fileInputRef.current.value = '';
                      }
                    }}
                  />
                  <input
                    ref={directoryInputRef}
                    type="file"
                    webkitDirectory
                    multiple
                    style={{ display: 'none' }}
                    onChange={async (e) => {
                      const files = e.target.files;
                      if (!files?.length) return;
                      const allowed = Array.from(files).filter(f => /\.(md|txt)$/i.test(f.name));
                      if (allowed.length === 0) {
                        alert('No .md or .txt files found in the selected folder.');
                        e.target.value = '';
                        return;
                      }
                      setFileUploadProgress(`Adding ${allowed.length} file(s)…`);
                      setEmbeddingsActionLoading(true);
                      try {
                        const form = new FormData();
                        allowed.forEach(f => form.append('files', f));
                        const r = await fetch('http://localhost:8000/api/rag/documents/files', { method: 'POST', body: form });
                        const data = await r.json();
                        if (!r.ok) throw new Error(data.detail || 'Upload failed');
                        setFileUploadProgress(`Added ${data.added} chunk(s) from ${data.files_processed} file(s).`);
                        setEmbeddingsChunkCount(prev => (prev != null ? prev + data.added : data.added));
                        if (ragStatus) setRagStatus(prev => prev ? { ...prev, document_count: (prev.document_count ?? 0) + data.added } : null);
                        fetch('http://localhost:8000/api/rag/chunks?limit=500&offset=0').then(res => res.json()).then(d => setEmbeddingsChunks(d.chunks || []));
                      } catch (err) {
                        alert(err.message || 'Upload failed');
                      } finally {
                        setEmbeddingsActionLoading(false);
                        e.target.value = '';
                        if (directoryInputRef.current) directoryInputRef.current.value = '';
                      }
                    }}
                  />
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={embeddingsActionLoading}
                      style={{
                        padding: '0.5rem 1rem', borderRadius: '8px', fontWeight: '600',
                        background: 'var(--accent-color)', color: 'white', border: 'none', cursor: embeddingsActionLoading ? 'not-allowed' : 'pointer',
                      }}
                    >
                      Choose files…
                    </button>
                    <button
                      type="button"
                      onClick={() => directoryInputRef.current?.click()}
                      disabled={embeddingsActionLoading}
                      style={{
                        padding: '0.5rem 1rem', borderRadius: '8px', fontWeight: '600',
                        background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border)', cursor: embeddingsActionLoading ? 'not-allowed' : 'pointer',
                      }}
                    >
                      Choose folder…
                    </button>
                  </div>
                  {fileUploadProgress && (
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>{fileUploadProgress}</div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
