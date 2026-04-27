"use client";

import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import {
  Button,
  Checkbox,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Tab,
  Tabs,
  TextField,
  Tooltip,
} from '@mui/material';

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

const markdownComponents = {
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
};

function splitThinkingSections(content) {
  const text = typeof content === 'string' ? content : '';
  const thoughtMatch = /(^|\n)\s*(Thinking|Reasoning):\s*\n?/i.exec(text);
  if (!thoughtMatch) return [{ type: 'answer', text }];

  const thoughtLabelStart = thoughtMatch.index + thoughtMatch[1].length;
  const thoughtContentStart = thoughtMatch.index + thoughtMatch[0].length;
  const answerRe = /(^|\n)\s*Answer:\s*\n?/i;
  answerRe.lastIndex = thoughtContentStart;
  const answerMatch = answerRe.exec(text.slice(thoughtContentStart));
  if (!answerMatch) {
    return [
      { type: 'answer', text: text.slice(0, thoughtLabelStart) },
      { type: 'thinking', label: thoughtMatch[2], text: text.slice(thoughtContentStart) },
    ].filter((part) => part.text);
  }

  const answerLabelStart = thoughtContentStart + answerMatch.index + answerMatch[1].length;
  const answerContentStart = thoughtContentStart + answerMatch.index + answerMatch[0].length;
  return [
    { type: 'answer', text: text.slice(0, thoughtLabelStart) },
    { type: 'thinking', label: thoughtMatch[2], text: text.slice(thoughtContentStart, answerLabelStart) },
    { type: 'answer', text: text.slice(answerContentStart) },
  ].filter((part) => part.text);
}

function MarkdownText({ text }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
      components={markdownComponents}
    >
      {preprocessMathForMarkdown(text)}
    </ReactMarkdown>
  );
}

function MessageContent({ content }) {
  return (
    <>
      {splitThinkingSections(content).map((part, idx) => {
        if (part.type !== 'thinking') {
          return <MarkdownText key={idx} text={part.text} />;
        }
        return (
          <div
            key={idx}
            style={{
              color: 'var(--text-secondary)',
              opacity: 0.72,
              borderLeft: '2px solid rgba(148, 163, 184, 0.35)',
              paddingLeft: '0.75rem',
              margin: '0.4rem 0',
            }}
          >
            <div style={{ fontSize: '0.78rem', fontWeight: 600, marginBottom: '0.25rem' }}>
              {part.label || 'Thinking'}
            </div>
            <MarkdownText text={part.text} />
          </div>
        );
      })}
    </>
  );
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

const muiFieldSx = {
  minWidth: 180,
  '& .MuiInputBase-root': {
    color: 'var(--text-primary)',
    backgroundColor: 'rgba(15, 23, 42, 0.55)',
    borderRadius: '10px',
  },
  '& .MuiInputLabel-root': { color: 'var(--text-secondary)' },
  '& .MuiOutlinedInput-notchedOutline': { borderColor: 'var(--border)' },
  '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(96, 165, 250, 0.65)' },
  '& .MuiSvgIcon-root': { color: 'var(--text-secondary)' },
};

const muiMenuProps = {
  PaperProps: {
    sx: {
      bgcolor: '#111827',
      color: 'var(--text-primary)',
      border: '1px solid var(--border)',
    },
  },
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
  /** Ollama: opt-in reasoning. Gemma 4 thought text is hidden, so default off keeps chat visibly streaming. */
  const [ollamaThinking, setOllamaThinking] = useState(false);
  /** OpenAI reasoning models: maps to LangChain ChatOpenAI reasoning_effort (none = omit). */
  const [openaiReasoning, setOpenaiReasoning] = useState('none');
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
  const [settingsTab, setSettingsTab] = useState('local'); // 'local' | 'cloud' | 'rag' | 'embeddings' | 'agents'
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

  // Insertion + education agents
  const [agentsConfig, setAgentsConfig] = useState(null);
  const [isInsertModalOpen, setIsInsertModalOpen] = useState(false);
  const [insertProgress, setInsertProgress] = useState(null);
  const [insertResults, setInsertResults] = useState([]);
  const [insertLoading, setInsertLoading] = useState(false);
  const [insertRejected, setInsertRejected] = useState([]);
  const [insertUrls, setInsertUrls] = useState('');
  const insertFileInputRef = useRef(null);

  const [isEduModalOpen, setIsEduModalOpen] = useState(false);
  const [eduTopic, setEduTopic] = useState('');
  const [eduAudience, setEduAudience] = useState('intermediate learners');
  const [eduFormat, setEduFormat] = useState('markdown');
  const [eduUseWeb, setEduUseWeb] = useState(true);
  const [eduExtraInstructions, setEduExtraInstructions] = useState('');
  const [eduLoading, setEduLoading] = useState(false);
  const [eduResult, setEduResult] = useState(null);
  const [eduError, setEduError] = useState(null);

  // Agent monitoring
  const [agentRuns, setAgentRuns] = useState([]);
  const [agentStats, setAgentStats] = useState(null);
  const [agentFilter, setAgentFilter] = useState('all'); // 'all' | 'insertion' | 'education'
  const [agentEvalDrafts, setAgentEvalDrafts] = useState({}); // runId -> {score, feedback}

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

  const loadAgentsConfig = useCallback(() => {
    fetch('http://localhost:8000/api/agents/config')
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setAgentsConfig(data); })
      .catch(() => {});
  }, []);

  const refreshAgentMonitoring = useCallback(() => {
    const q = agentFilter === 'all' ? '' : `?agent=${agentFilter}`;
    fetch(`http://localhost:8000/api/agents/runs${q}${q ? '&' : '?'}limit=50`)
      .then(r => r.ok ? r.json() : { runs: [] })
      .then(data => setAgentRuns(data.runs || []))
      .catch(() => setAgentRuns([]));
    fetch(`http://localhost:8000/api/agents/stats${agentFilter === 'all' ? '' : `?agent=${agentFilter}`}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setAgentStats(data))
      .catch(() => setAgentStats(null));
  }, [agentFilter]);

  useEffect(() => { loadAgentsConfig(); }, [loadAgentsConfig]);

  useEffect(() => {
    if (!(isManageModalOpen && settingsTab === 'agents')) return;
    refreshAgentMonitoring();
    const id = setInterval(refreshAgentMonitoring, 4000);
    return () => clearInterval(id);
  }, [isManageModalOpen, settingsTab, refreshAgentMonitoring]);

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
            thinking: ollamaThinking,
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
            ...(provider === 'openai'
              ? { reasoning: openaiReasoning }
              : {}),
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
            ...(provider === 'openai' ? { reasoning: openaiReasoning } : {}),
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

        if (!response.body) {
          throw new Error('Streaming response body is not available in this browser.');
        }
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
        const trailingText = decoder.decode();
        if (trailingText) {
          setMessages((prev) => {
            const allExceptLast = prev.slice(0, prev.length - 1);
            const lastMsg = prev[prev.length - 1];
            return [
              ...allExceptLast,
              { ...lastMsg, content: lastMsg.content + trailingText },
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

  const allowedInsertExts = agentsConfig?.allowed_insertion_extensions || [
    '.pdf', '.txt', '.md', '.jpg', '.jpeg', '.png', '.heic', '.json',
    '.html', '.htm', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
  ];

  const handleInsertFiles = async (fileList) => {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;
    const accepted = [];
    const rejected = [];
    for (const f of files) {
      const lower = (f.name || '').toLowerCase();
      if (allowedInsertExts.some(ext => lower.endsWith(ext))) accepted.push(f);
      else rejected.push(f.name);
    }
    setInsertRejected(rejected);
    if (accepted.length === 0) {
      setInsertProgress('No allowed files selected.');
      return;
    }
    setInsertLoading(true);
    setInsertProgress(`Uploading ${accepted.length} file(s)…`);
    try {
      const form = new FormData();
      for (const f of accepted) form.append('files', f);
      const r = await fetch('http://localhost:8000/api/agents/insertion/upload', {
        method: 'POST',
        body: form,
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail?.message || data?.detail || r.statusText);
      setInsertResults(data.results || []);
      setInsertRejected(prev => Array.from(new Set([...(prev || []), ...(data.rejected || [])])));
      const stored = (data.results || []).reduce((acc, x) => acc + (x.stored_chunks || 0), 0);
      setInsertProgress(`Processed ${data.processed || 0} file(s); stored ${stored} chunk(s).`);
    } catch (err) {
      setInsertProgress(`Error: ${err.message || err}`);
    } finally {
      setInsertLoading(false);
      if (insertFileInputRef.current) insertFileInputRef.current.value = '';
    }
  };

  const handleInsertUrls = async () => {
    const urls = insertUrls
      .split(/\s+/)
      .map((u) => u.trim())
      .filter(Boolean);
    if (urls.length === 0) {
      setInsertProgress('Enter at least one web link.');
      return;
    }
    setInsertLoading(true);
    setInsertProgress(`Fetching ${urls.length} link(s)…`);
    setInsertRejected([]);
    try {
      const r = await fetch('http://localhost:8000/api/agents/insertion/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail?.message || data?.detail || r.statusText);
      setInsertResults(data.results || []);
      setInsertRejected(data.rejected || []);
      const stored = (data.results || []).reduce((acc, x) => acc + (x.stored_chunks || 0), 0);
      setInsertProgress(`Inserted ${data.processed || 0} link(s); stored ${stored} chunk(s).`);
    } catch (err) {
      setInsertProgress(`Error: ${err.message || err}`);
    } finally {
      setInsertLoading(false);
    }
  };

  const openInsertedDataView = () => {
    setSettingsTab('embeddings');
    setIsManageModalOpen(true);
    setEmbeddingsError(null);
    setChunksLoading(true);
    fetch('http://localhost:8000/api/rag/chunks?limit=500&offset=0')
      .then(r => {
        if (!r.ok) return r.json().then(data => { setEmbeddingsError(data.detail || r.statusText); setEmbeddingsChunks([]); });
        return r.json().then(data => { setEmbeddingsChunks(data.chunks || []); setEmbeddingsError(null); });
      })
      .catch(() => { setEmbeddingsChunks([]); setEmbeddingsError('Failed to load chunks'); })
      .finally(() => setChunksLoading(false));
  };

  const handleEducationGenerate = async () => {
    if (!eduTopic.trim()) {
      setEduError('Please enter a topic.');
      return;
    }
    setEduError(null);
    setEduLoading(true);
    setEduResult(null);
    try {
      const r = await fetch('http://localhost:8000/api/agents/education/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: eduTopic.trim(),
          audience: eduAudience.trim() || 'intermediate learners',
          format: eduFormat,
          use_web: eduUseWeb,
          extra_instructions: eduExtraInstructions.trim(),
        }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail || r.statusText);
      setEduResult(data);
    } catch (err) {
      setEduError(err.message || String(err));
    } finally {
      setEduLoading(false);
    }
  };

  const submitAgentEvaluation = async (run) => {
    const draft = agentEvalDrafts[run.id] || {};
    const score = draft.score === '' || draft.score === undefined ? null : Number(draft.score);
    const feedback = (draft.feedback || '').trim();
    if (score === null && !feedback) return;
    try {
      const r = await fetch('http://localhost:8000/api/agents/runs/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent: run.agent, run_id: run.id, score, feedback }),
      });
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        throw new Error(data?.detail || r.statusText);
      }
      refreshAgentMonitoring();
      setAgentEvalDrafts(prev => ({ ...prev, [run.id]: { score: '', feedback: '' } }));
    } catch (err) {
      alert(`Failed to save evaluation: ${err.message || err}`);
    }
  };

  return (
    <>
      <main className="container">
        {/* Header */}
        <header className="header" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', position: 'relative' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="header-title" style={{ flex: 1, textAlign: 'center' }}>
              <span>♇</span> Pluto
            </div>

            <div style={{ position: 'absolute', top: '10px', right: '15px', display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
              <Button
                onClick={() => { setIsInsertModalOpen(true); setInsertProgress(null); setInsertResults([]); setInsertRejected([]); setInsertUrls(''); }}
                variant="contained"
                size="small"
                sx={{ textTransform: 'none', borderRadius: '10px', fontWeight: 700 }}
                title="Insert documents into the knowledge base"
              >
                Insert
              </Button>
              <Button
                onClick={() => { setIsEduModalOpen(true); setEduResult(null); setEduError(null); }}
                variant="contained"
                color="success"
                size="small"
                sx={{ textTransform: 'none', borderRadius: '10px', fontWeight: 700 }}
                title="Generate educational content"
              >
                Generate
              </Button>
              <Button
                onClick={openInsertedDataView}
                variant="outlined"
                size="small"
                sx={{
                  textTransform: 'none',
                  borderRadius: '10px',
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  borderColor: 'var(--border)',
                }}
                title="View inserted chunks"
              >
                View
              </Button>
              <Tooltip title="Settings">
                <IconButton
                  onClick={() => setIsManageModalOpen(true)}
                  size="small"
                  sx={{ color: 'var(--text-secondary)' }}
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                  </svg>
                </IconButton>
              </Tooltip>
            </div>
          </div>

          {/* Provider Tabs */}
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: '0.2rem' }}>
            <Tabs
              value={provider}
              onChange={(_, key) => {
                setProvider(key);
                if (key === 'ollama') refreshOllamaModelState();
              }}
              variant="scrollable"
              sx={{
                minHeight: 36,
                '& .MuiTab-root': {
                  color: 'var(--text-secondary)',
                  minHeight: 36,
                  textTransform: 'none',
                  fontWeight: 700,
                },
                '& .Mui-selected': { color: 'var(--text-primary)' },
                '& .MuiTabs-indicator': { backgroundColor: 'var(--accent-color, #60a5fa)' },
              }}
            >
              {Object.entries(PROVIDER_LABELS).map(([key, label]) => (
                <Tab key={key} value={key} label={label} />
              ))}
            </Tabs>
          </div>

          {/* Model Selector */}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <Stack
              direction="row"
              spacing={1}
              alignItems="center"
              sx={{ background: 'var(--bg-secondary)', p: 0.75, borderRadius: '12px', border: '1px solid var(--border)' }}
            >
              <Button
                type="button"
                onClick={() => { setMessages([]); setThreadId(typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `thread-${Date.now()}`); setError(null); }}
                variant="text"
                size="small"
                sx={{ color: 'var(--text-secondary)', textTransform: 'none', fontWeight: 700 }}
                title="Start a new conversation (new thread)"
              >
                New chat
              </Button>
              {provider === 'ollama' ? (
                <FormControl size="small" sx={muiFieldSx}>
                  <InputLabel id="local-model-label">Model</InputLabel>
                  <Select
                    labelId="local-model-label"
                    label="Model"
                    value={selectedModel}
                    onChange={e => setSelectedModel(e.target.value)}
                    MenuProps={muiMenuProps}
                  >
                    {loadedModels.length === 0 && <MenuItem value="none">No models loaded</MenuItem>}
                    {loadedModels.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
                  </Select>
                </FormControl>
              ) : (
                <FormControl size="small" sx={muiFieldSx}>
                  <InputLabel id="cloud-model-label">Model</InputLabel>
                  <Select
                    labelId="cloud-model-label"
                    label="Model"
                    value={cloudModels[provider]}
                    onChange={e => setCloudModels(prev => ({ ...prev, [provider]: e.target.value }))}
                    MenuProps={muiMenuProps}
                  >
                    {CLOUD_MODELS[provider].map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
                  </Select>
                </FormControl>
              )}
              {/* Google requires a browser API key; OpenAI/Anthropic use server .env.local */}
              {provider === 'google' && !apiKeys[provider] && (
                <span title="API key not set" style={{ color: '#ef4444', fontSize: '0.7rem', cursor: 'help' }}>⚠</span>
              )}
            </Stack>
          </div>
          {provider === 'ollama' && (
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '0.35rem' }}>
              <FormControlLabel
                control={
                  <Checkbox
                    size="small"
                    checked={ollamaThinking}
                    onChange={(e) => setOllamaThinking(e.target.checked)}
                    sx={{ color: 'var(--text-secondary)' }}
                />
                }
                label="Thinking / reasoning (off by default; Gemma 4 hides thought text before streaming the answer)"
                sx={{
                  color: 'var(--text-secondary)',
                  '& .MuiFormControlLabel-label': { fontSize: '0.78rem' },
                }}
              />
            </div>
          )}
          {provider === 'openai' && (
            <Stack direction="row" justifyContent="center" alignItems="center" sx={{ mt: 0.5 }}>
              <FormControl size="small" sx={{ ...muiFieldSx, minWidth: 160 }}>
                <InputLabel id="openai-reasoning-label">Reasoning effort</InputLabel>
                <Select
                  labelId="openai-reasoning-label"
                  label="Reasoning effort"
                  value={openaiReasoning}
                  onChange={(e) => setOpenaiReasoning(e.target.value)}
                  MenuProps={muiMenuProps}
                >
                  <MenuItem value="none">none (default)</MenuItem>
                  <MenuItem value="low">low</MenuItem>
                  <MenuItem value="medium">medium</MenuItem>
                  <MenuItem value="high">high</MenuItem>
                  <MenuItem value="xhigh">xhigh</MenuItem>
                </Select>
              </FormControl>
            </Stack>
          )}
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
              <h2>Welcome to Pluto!</h2>
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
                <MessageContent content={msg.content} />
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
            <TextField
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              multiline
              minRows={1}
              maxRows={8}
              disabled={isLoading}
              fullWidth
              variant="outlined"
              size="small"
              sx={{
                '& .MuiInputBase-root': {
                  color: 'var(--text-primary)',
                  background: 'var(--input-bg, rgba(15, 23, 42, 0.72))',
                  borderRadius: '18px',
                  pr: 0.5,
                },
                '& .MuiOutlinedInput-notchedOutline': { borderColor: 'var(--border)' },
                '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(96, 165, 250, 0.65)' },
              }}
            />
            <IconButton
              type="submit"
              disabled={!isLoading && !input.trim()}
              aria-label={isLoading ? "Stop response" : "Send Message"}
              sx={{
                width: 44,
                height: 44,
                ml: 1,
                color: 'white',
                backgroundColor: isLoading ? '#ef4444' : 'var(--accent-color, #2563eb)',
                '&:hover': { backgroundColor: isLoading ? '#dc2626' : '#1d4ed8' },
                '&.Mui-disabled': {
                  backgroundColor: 'var(--bg-secondary)',
                  color: 'var(--text-secondary)',
                },
              }}
            >
              {isLoading ? (
                <svg className="send-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                  <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
                </svg>
              ) : (
                <svg className="send-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                  <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </IconButton>
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
            <Tabs
              value={settingsTab}
              onChange={(_, value) => setSettingsTab(value)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{
                mb: 2,
                '& .MuiTab-root': {
                  color: 'var(--text-secondary)',
                  minHeight: 38,
                  textTransform: 'none',
                  fontWeight: 700,
                },
                '& .Mui-selected': { color: 'var(--text-primary)' },
                '& .MuiTabs-indicator': { backgroundColor: 'var(--accent-color, #60a5fa)' },
              }}
            >
              <Tab value="local" label="Local Models" />
              <Tab value="cloud" label="Cloud API Keys" />
              <Tab value="rag" label="Knowledge base" />
              <Tab value="embeddings" label="Embeddings & storage" />
              <Tab value="agents" label="Agents" />
            </Tabs>

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

            {/* Agents monitoring / evaluation Tab */}
            {settingsTab === 'agents' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Monitor and evaluate the insertion and education agents. Models are configured in <code>backend/settings.json</code>.
                </p>

                {agentsConfig && (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', background: 'var(--bg-secondary)', padding: '0.5rem 0.75rem', borderRadius: '6px', border: '1px solid var(--border)' }}>
                    Insertion: <strong>{agentsConfig.insertion_agent_model}</strong> · Education: <strong>{agentsConfig.education_agent_model}</strong> · Embeddings: <strong>{agentsConfig.embedding_model}</strong>
                  </div>
                )}

                <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Filter:</label>
                  {['all', 'insertion', 'education'].map(key => (
                    <button
                      key={key}
                      onClick={() => setAgentFilter(key)}
                      style={{
                        padding: '0.3rem 0.7rem', borderRadius: '6px', fontSize: '0.8rem',
                        background: agentFilter === key ? 'var(--accent-color, #2563eb)' : 'var(--bg-secondary)',
                        color: agentFilter === key ? 'white' : 'var(--text-primary)',
                        border: '1px solid var(--border)', cursor: 'pointer', fontWeight: 600,
                      }}
                    >
                      {key}
                    </button>
                  ))}
                  <button
                    onClick={refreshAgentMonitoring}
                    style={{ marginLeft: 'auto', padding: '0.3rem 0.7rem', borderRadius: '6px', fontSize: '0.8rem', background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border)', cursor: 'pointer' }}
                  >
                    Refresh
                  </button>
                </div>

                {agentStats && (
                  <div style={{ display: 'grid', gridTemplateColumns: agentFilter === 'all' ? '1fr 1fr' : '1fr', gap: '0.5rem' }}>
                    {agentFilter === 'all' ? (
                      <>
                        <AgentStatsCard title="Insertion" stats={agentStats.insertion} />
                        <AgentStatsCard title="Education" stats={agentStats.education} />
                      </>
                    ) : (
                      <AgentStatsCard title={agentFilter} stats={agentStats} />
                    )}
                  </div>
                )}

                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem' }}>Recent runs</div>
                  {agentRuns.length === 0 ? (
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>No runs yet. Use the Insert or Generate buttons to create activity.</div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '340px', overflowY: 'auto', border: '1px solid var(--border)', borderRadius: '8px', padding: '0.4rem' }}>
                      {agentRuns.map(run => {
                        const draft = agentEvalDrafts[run.id] || { score: '', feedback: '' };
                        const statusColor = run.status === 'ok' ? '#10b981' : run.status === 'error' ? '#ef4444' : '#f59e0b';
                        return (
                          <div key={run.id} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '6px', padding: '0.5rem 0.6rem' }}>
                            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.78rem' }}>
                              <span style={{ color: statusColor, fontWeight: 700 }}>●</span>
                              <span style={{ fontWeight: 600 }}>{run.agent}</span>
                              <span style={{ color: 'var(--text-secondary)' }}>{run.action}</span>
                              <span style={{ color: 'var(--text-secondary)', marginLeft: 'auto' }}>{run.duration_s != null ? `${run.duration_s}s` : run.status}</span>
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                              {run.started_at} · {(run.metadata?.filename) || run.metadata?.topic || run.metadata?.model || ''}
                            </div>
                            {run.metrics && Object.keys(run.metrics).length > 0 && (
                              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                                {Object.entries(run.metrics).map(([k, v]) => (
                                  <span key={k} style={{ marginRight: '0.6rem' }}>
                                    <strong>{k}:</strong> {typeof v === 'string' ? v.slice(0, 80) : String(v)}
                                  </span>
                                ))}
                              </div>
                            )}
                            {run.error && (
                              <div style={{ fontSize: '0.75rem', color: '#fca5a5', marginTop: '0.2rem' }}>Error: {run.error}</div>
                            )}
                            {run.evaluation && (
                              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                                Eval: score {run.evaluation.score ?? '—'}{run.evaluation.feedback ? ` · ${run.evaluation.feedback}` : ''}
                              </div>
                            )}
                            <div style={{ display: 'flex', gap: '0.3rem', marginTop: '0.4rem', alignItems: 'center' }}>
                              <input
                                type="number" min="0" max="5" step="0.5"
                                placeholder="score 0-5"
                                value={draft.score}
                                onChange={e => setAgentEvalDrafts(prev => ({ ...prev, [run.id]: { ...(prev[run.id] || {}), score: e.target.value } }))}
                                style={{ width: '85px', padding: '0.2rem 0.4rem', fontSize: '0.75rem', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderRadius: '4px', color: 'var(--text-primary)' }}
                              />
                              <input
                                type="text"
                                placeholder="feedback"
                                value={draft.feedback}
                                onChange={e => setAgentEvalDrafts(prev => ({ ...prev, [run.id]: { ...(prev[run.id] || {}), feedback: e.target.value } }))}
                                style={{ flex: 1, padding: '0.2rem 0.4rem', fontSize: '0.75rem', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderRadius: '4px', color: 'var(--text-primary)' }}
                              />
                              <button
                                onClick={() => submitAgentEvaluation(run)}
                                style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem', background: 'var(--accent-color, #2563eb)', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 600 }}
                              >
                                Save
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Insertion Modal */}
      {isInsertModalOpen && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(4px)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 9999 }}>
          <div style={{ backgroundColor: '#161b22', padding: '2rem', borderRadius: '12px', minWidth: '460px', maxWidth: '560px', border: '1px solid var(--chat-border)', boxShadow: '0 10px 25px rgba(0,0,0,0.8)', color: 'var(--text-primary)', maxHeight: '80vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 style={{ margin: 0, fontSize: '1.15rem' }}>Insert documents</h2>
              <button onClick={() => setIsInsertModalOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '1.2rem' }}>✕</button>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: 0 }}>
              The insertion agent stores files and web pages in <code>data/uploads</code> and embeds them into ChromaDB using <strong>{agentsConfig?.insertion_agent_model || 'the configured insertion model'}</strong>. Other file types are rejected.
            </p>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
              Allowed: {allowedInsertExts.join(', ')}
            </p>

            <input
              ref={insertFileInputRef}
              type="file"
              multiple
              accept={allowedInsertExts.join(',')}
              style={{ display: 'none' }}
              onChange={(e) => handleInsertFiles(e.target.files)}
            />
            <button
              onClick={() => insertFileInputRef.current?.click()}
              disabled={insertLoading}
              style={{ padding: '0.5rem 1rem', borderRadius: '8px', fontWeight: 600, background: insertLoading ? 'var(--bg-secondary)' : 'var(--accent-color, #2563eb)', color: 'white', border: 'none', cursor: insertLoading ? 'not-allowed' : 'pointer' }}
            >
              {insertLoading ? 'Uploading…' : 'Choose files…'}
            </button>

            <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
              <label style={{ display: 'block', fontSize: '0.86rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                Insert web links
              </label>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '0 0 0.5rem 0' }}>
                Add one URL per line or separated by spaces. PDF files and HTML pages are supported.
              </p>
              <textarea
                className="chat-input"
                value={insertUrls}
                onChange={(e) => setInsertUrls(e.target.value)}
                placeholder="https://example.com/article.html&#10;https://example.com/paper.pdf"
                rows={3}
                disabled={insertLoading}
                style={{ width: '100%', resize: 'vertical' }}
              />
              <button
                type="button"
                onClick={handleInsertUrls}
                disabled={insertLoading || !insertUrls.trim()}
                style={{
                  marginTop: '0.5rem',
                  padding: '0.5rem 1rem',
                  borderRadius: '8px',
                  fontWeight: 600,
                  background: insertLoading || !insertUrls.trim() ? 'var(--bg-secondary)' : '#10b981',
                  color: insertLoading || !insertUrls.trim() ? 'var(--text-secondary)' : 'white',
                  border: '1px solid var(--border)',
                  cursor: insertLoading || !insertUrls.trim() ? 'not-allowed' : 'pointer',
                }}
              >
                {insertLoading ? 'Inserting…' : 'Insert links'}
              </button>
            </div>

            {insertProgress && (
              <div style={{ fontSize: '0.85rem', marginTop: '0.75rem', color: 'var(--text-secondary)' }}>{insertProgress}</div>
            )}
            {insertRejected.length > 0 && (
              <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#fca5a5' }}>
                Rejected: {insertRejected.join(', ')}
              </div>
            )}
            {insertResults.length > 0 && (
              <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {insertResults.map((r, i) => (
                  <div key={i} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '6px', padding: '0.5rem 0.6rem', fontSize: '0.82rem' }}>
                    <div style={{ fontWeight: 600 }}>{r.file}</div>
                    {r.source_url && <div style={{ color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{r.source_url}</div>}
                    {r.error ? (
                      <div style={{ color: '#fca5a5' }}>Error: {r.error}</div>
                    ) : (
                      <>
                        <div style={{ color: 'var(--text-secondary)' }}>Stored chunks: {r.stored_chunks ?? 0}{r.fallback ? ' (fallback)' : ''}</div>
                        {r.agent_output && <div style={{ color: 'var(--text-secondary)', marginTop: '0.2rem' }}>{r.agent_output.slice(0, 260)}</div>}
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Education Modal */}
      {isEduModalOpen && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(4px)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 9999 }}>
          <div style={{ backgroundColor: '#161b22', padding: '2rem', borderRadius: '12px', minWidth: '520px', maxWidth: '620px', border: '1px solid var(--chat-border)', boxShadow: '0 10px 25px rgba(0,0,0,0.8)', color: 'var(--text-primary)', maxHeight: '85vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 style={{ margin: 0, fontSize: '1.15rem' }}>Generate educational content</h2>
              <button onClick={() => setIsEduModalOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '1.2rem' }}>✕</button>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: 0 }}>
              Education agent: <strong>{agentsConfig?.education_agent_model || 'configured'}</strong> with RAG over the local KB (embeddings: <strong>{agentsConfig?.embedding_model || 'configured'}</strong>) and optional web search.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.7rem' }}>
              <div>
                <TextField
                  label="Topic"
                  value={eduTopic}
                  onChange={e => setEduTopic(e.target.value)}
                  placeholder="e.g. Introduction to Kalman filters"
                  fullWidth
                  size="small"
                  sx={muiFieldSx}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem' }}>
                <div>
                  <TextField
                    label="Audience"
                    value={eduAudience}
                    onChange={e => setEduAudience(e.target.value)}
                    fullWidth
                    size="small"
                    sx={muiFieldSx}
                  />
                </div>
                <div>
                  <FormControl fullWidth size="small" sx={muiFieldSx}>
                    <InputLabel id="education-format-label">Format</InputLabel>
                    <Select
                      labelId="education-format-label"
                      label="Format"
                      value={eduFormat}
                      onChange={e => setEduFormat(e.target.value)}
                      MenuProps={muiMenuProps}
                    >
                      <MenuItem value="markdown">markdown</MenuItem>
                      <MenuItem value="html">html</MenuItem>
                      <MenuItem value="json">json</MenuItem>
                      <MenuItem value="pdf">pdf</MenuItem>
                    </Select>
                  </FormControl>
                </div>
              </div>
              <div>
                <TextField
                  label="Extra instructions (optional)"
                  value={eduExtraInstructions}
                  onChange={e => setEduExtraInstructions(e.target.value)}
                  minRows={2}
                  multiline
                  fullWidth
                  size="small"
                  sx={muiFieldSx}
                />
              </div>
              <FormControlLabel
                control={<Checkbox size="small" checked={eduUseWeb} onChange={e => setEduUseWeb(e.target.checked)} sx={{ color: 'var(--text-secondary)' }} />}
                label="Enable web search (DuckDuckGo)"
                sx={{ color: 'var(--text-secondary)', '& .MuiFormControlLabel-label': { fontSize: '0.82rem' } }}
              />

              <Stack direction="row" spacing={1}>
                <Button
                  onClick={handleEducationGenerate}
                  disabled={eduLoading || !eduTopic.trim()}
                  variant="contained"
                  color="success"
                  sx={{ flex: 1, borderRadius: '10px', textTransform: 'none', fontWeight: 700 }}
                >
                  {eduLoading ? 'Generating… (this may take a minute)' : 'Generate'}
                </Button>
                <Button
                  onClick={openInsertedDataView}
                  variant="outlined"
                  sx={{
                    borderRadius: '10px',
                    textTransform: 'none',
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                    borderColor: 'var(--border)',
                  }}
                >
                  View
                </Button>
              </Stack>

              {eduError && (
                <div style={{ fontSize: '0.85rem', color: '#fca5a5' }}>Error: {eduError}</div>
              )}
              {eduResult && (
                <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '8px', padding: '0.6rem 0.75rem', fontSize: '0.85rem' }}>
                  <div style={{ fontWeight: 600 }}>Result ({eduResult.format})</div>
                  {eduResult.output_path ? (
                    <div style={{ color: 'var(--text-secondary)', margin: '0.25rem 0' }}>
                      Saved to: <code>{eduResult.output_path}</code>{' '}
                      <a
                        href={`http://localhost:8000/api/agents/education/download?path=${encodeURIComponent(eduResult.output_path)}`}
                        target="_blank"
                        rel="noreferrer"
                        style={{ color: 'var(--accent-color, #60a5fa)', marginLeft: '0.35rem' }}
                      >
                        Download
                      </a>
                    </div>
                  ) : (
                    <div style={{ color: '#fca5a5' }}>The agent did not persist an output file. See agent message below.</div>
                  )}
                  {eduResult.plan && (
                    <details style={{ marginTop: '0.4rem' }}>
                      <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>Plan (chain-of-thought)</summary>
                      <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.78rem', marginTop: '0.3rem' }}>{eduResult.plan}</pre>
                    </details>
                  )}
                  {eduResult.agent_output && (
                    <details style={{ marginTop: '0.4rem' }}>
                      <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>Agent message</summary>
                      <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.78rem', marginTop: '0.3rem' }}>{eduResult.agent_output}</pre>
                    </details>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function AgentStatsCard({ title, stats }) {
  if (!stats) return null;
  return (
    <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '8px', padding: '0.6rem 0.75rem', fontSize: '0.8rem' }}>
      <div style={{ fontWeight: 700, textTransform: 'capitalize', marginBottom: '0.2rem' }}>{title}</div>
      <div style={{ color: 'var(--text-secondary)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.15rem 0.6rem' }}>
        <span>Total</span><span>{stats.total_runs ?? 0}</span>
        <span>OK</span><span style={{ color: '#10b981' }}>{stats.ok_runs ?? 0}</span>
        <span>Errors</span><span style={{ color: '#ef4444' }}>{stats.error_runs ?? 0}</span>
        <span>Running</span><span style={{ color: '#f59e0b' }}>{stats.running_runs ?? 0}</span>
        <span>Success rate</span><span>{stats.success_rate ?? 0}%</span>
        <span>Avg duration</span><span>{stats.avg_duration_s ?? 0}s</span>
      </div>
    </div>
  );
}
