// js/ai-assistant-ux.js — AI panel overlay con chat, suggestions, resize
import { apiFetch, apiBase } from './api.js';
import * as state from './state.js';
import { makeResizable } from './make-resizable.js';
import { applyOverridesToUI } from './params.js';

const AI_SUGGESTIONS = [
  '¿Cómo suena esto?',
  '¿Qué le falta?',
  'Masteriza esto por mí',
  '¿Está listo para streaming?',
];
let chatHistory = [];
let aiAvailable = null;
let panelOpen = false;
let wired = false;

function emit(name, detail) {
  window.dispatchEvent(new CustomEvent(`lgmdm:${name}`, { detail }));
}

function buildPanel() {
  if (document.getElementById('aiPanel')) return;
  const panel = document.createElement('div');
  panel.id = 'aiPanel';
  panel.className = 'ai-panel hidden';
  panel.innerHTML = `
    <div id="aiPanelResizeHandle" class="ai-resize-handle" tabindex="0"></div>
    <div class="ai-panel-header">
      <span id="aiStatusLine">Laia — Asistente de Mastering</span>
      <button id="aiClose">✕</button>
    </div>
    <div id="aiMessages" class="ai-messages"></div>
    <div id="aiSuggestions" class="ai-suggestions"></div>
    <div class="ai-input-row">
      <textarea id="aiInput" placeholder="Preguntale a Laia…" rows="1"></textarea>
      <button id="aiSend">▶</button>
    </div>
  `;
  document.body.appendChild(panel);

  document.getElementById('aiClose')?.addEventListener('click', () => {
    panelOpen = false;
    panel.classList.add('hidden');
  });
  document.getElementById('aiSend')?.addEventListener('click', sendMessage);
  document.getElementById('aiInput')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  document.getElementById('aiInput')?.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 96) + 'px';
  });

  makeResizable(document.getElementById('aiPanelResizeHandle'), {
    axis: 'x', invert: true,
    getSize: () => panel.offsetWidth,
    setSize: (v) => { panel.style.setProperty('--ai-panel-w', v + 'px'); panel.style.width = v + 'px'; },
    min: 300, max: 600,
  });
  makeResizable(panel, {
    axis: 'y', invert: true,
    getSize: () => panel.offsetHeight,
    setSize: (v) => { panel.style.setProperty('--ai-panel-h', v + 'px'); panel.style.height = v + 'px'; },
    min: 260, max: 700,
  });
}

function appendMessage(role, content) {
  const el = document.getElementById('aiMessages');
  if (!el) return;
  const div = document.createElement('div');
  div.className = `ai-msg ${role}`;
  div.textContent = content;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return div;
}

function showTyping() {
  return appendMessage('assistant typing', '…');
}

function hideTyping() {
  const el = document.getElementById('aiMessages');
  const typing = el?.querySelector('.typing');
  if (typing) typing.remove();
}

function renderSuggestions() {
  const el = document.getElementById('aiSuggestions');
  if (!el) return;
  el.innerHTML = AI_SUGGESTIONS.map(s =>
    `<button class="ai-suggestion-btn">${s}</button>`
  ).join('');
  el.querySelectorAll('.ai-suggestion-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const text = btn.textContent;
      if (text === 'Masteriza esto por mí') {
        document.getElementById('btn-auto-master')?.click();
        return;
      }
      document.getElementById('aiInput').value = text;
      sendMessage();
    });
  });
}

function appendSuggestionCard(suggestedParams, summary, explanation) {
  const el = document.getElementById('aiMessages');
  if (!el) return;
  const card = document.createElement('div');
  card.className = 'ai-msg ai-suggestion-card';
  const paramList = Object.entries(suggestedParams || {}).slice(0, 8)
    .map(([k, v]) => `<div class="ai-suggestion-param"><span>${k}</span><span>${typeof v === 'number' ? v.toFixed(2) : v}</span></div>`)
    .join('');
  card.innerHTML = `
    <div class="ai-suggestion-card-title">${summary || 'Sugerencia de IA'}</div>
    <div class="ai-suggestion-explanation">${explanation || ''}</div>
    <div class="ai-suggestion-card-list">${paramList}</div>
    <div class="ai-suggestion-card-actions">
      <button class="ai-suggestion-apply-btn">✓ Aplicar</button>
      <button class="ai-suggestion-cancel-btn">✕ Cancelar</button>
    </div>
  `;
  card.querySelector('.ai-suggestion-apply-btn')?.addEventListener('click', () => {
    state.set('suggestedParams', suggestedParams);
    emit('ai-apply-suggestion', { params: suggestedParams });
    card.classList.add('applied');
  });
  card.querySelector('.ai-suggestion-cancel-btn')?.addEventListener('click', () => card.remove());
  el.appendChild(card);
  el.scrollTop = el.scrollHeight;
}

async function checkStatus() {
  try {
    const res = await apiFetch(`${apiBase()}/ai/status`);
    if (!res.ok) { aiAvailable = false; return; }
    const data = await res.json();
    aiAvailable = data.available === true;
    const line = document.getElementById('aiStatusLine');
    if (line) line.textContent = aiAvailable ? 'Laia — Asistente de Mastering' : `Laia — ${data.reason || 'No disponible'}`;
  } catch { aiAvailable = false; }
}

async function sendMessage() {
  const input = document.getElementById('aiInput');
  const msg = input?.value?.trim();
  if (!msg) return;
  input.value = '';
  appendMessage('user', msg);
  chatHistory.push({ role: 'user', content: msg });

  if (aiAvailable === false) {
    appendMessage('system-note', 'La IA no está configurada. Necesitás configurar GEMINI_API_KEY.');
    return;
  }
  if (aiAvailable === null) await checkStatus();
  if (!aiAvailable) return;

  const typing = showTyping();
  try {
    const ctx = state.get('lastAnalysis') || {};
    const preset = document.querySelector('.preset-btn.active')?.dataset?.preset || '';
    const platform = document.getElementById('s-platform')?.value || '';
    const res = await apiFetch(`${apiBase()}/ai/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, history: chatHistory.slice(-20), analysis: ctx, preset, platform }),
    });
    hideTyping();
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    appendMessage('assistant', data.reply);
    chatHistory.push({ role: 'assistant', content: data.reply });
    if (data.suggested_params) {
      appendSuggestionCard(data.suggested_params, data.suggestion_summary, data.suggestion_explanation);
    }
  } catch (err) {
    hideTyping();
    appendMessage('system-note', `Error: ${err.message}`);
  }
}

function togglePanel() {
  buildPanel();
  const panel = document.getElementById('aiPanel');
  if (!panel) return;
  panelOpen = !panelOpen;
  panel.classList.toggle('hidden', !panelOpen);
  if (panelOpen) {
    renderSuggestions();
    checkStatus();
    if (chatHistory.length === 0) {
      appendMessage('assistant', '¡Hola! Soy Laia, tu asistente de mastering. ¿En qué te puedo ayudar?');
    }
  }
}

export function setContext(analysisData) {
  state.set('lastAnalysis', analysisData);
  const fab = document.getElementById('aiFab');
  if (fab && analysisData) fab.classList.add('has-context');
  emit('analysis-updated', analysisData);
}

export function init() {
  if (init.done) return;
  init.done = true;

  document.getElementById('aiFab')?.addEventListener('click', togglePanel);

  window.addEventListener('lgmdm:ai-apply-suggestion', (e) => {
    const params = e.detail?.params;
    if (!params || typeof params !== 'object') return;
    applyOverridesToUI(params);
    state.set('suggestedParams', params);
    emit('analysis-updated', params);
    emit('param-change', { source: 'ai' });
  });
}
