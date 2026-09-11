// js/reference-library-picker.js — Modal de biblioteca de referencias
import { apiFetch, apiBase, getToken } from './api.js';
import * as state from './state.js';

let entries = [];
let filtered = [];
let selected = null;
let modalBuilt = false;

function buildModal() {
  if (modalBuilt) return;
  modalBuilt = true;
  const backdrop = document.createElement('div');
  backdrop.id = 'refLibModal';
  backdrop.className = 'ref-lib-backdrop';
  backdrop.style.display = 'none';
  backdrop.innerHTML = `
    <div class="ref-lib-dialog">
      <div class="ref-lib-header">
        <h3>Biblioteca de Referencias</h3>
        <button id="refLibClose" class="ref-lib-close">✕</button>
      </div>
      <div class="ref-lib-toolbar">
        <input type="text" id="refLibSearch" placeholder="Buscar por nombre…">
        <button id="refLibRescan" class="btn btn-secondary btn-sm">🔄 Re-escanear</button>
      </div>
      <div id="refLibStatus" class="ref-lib-status"></div>
      <div id="refLibList" class="ref-lib-list"></div>
      <div class="ref-lib-footer">Seleccioná una referencia para usar en el mastering</div>
    </div>
  `;
  document.body.appendChild(backdrop);

  backdrop.querySelector('#refLibClose')?.addEventListener('click', closeModal);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) closeModal(); });
  backdrop.querySelector('#refLibSearch')?.addEventListener('input', (e) => applySearch(e.target.value));
  backdrop.querySelector('#refLibRescan')?.addEventListener('click', async () => {
    try {
      await apiFetch(`${apiBase()}/reference-library/rescan`, { method: 'POST' });
      await reload();
    } catch {}
  });
}

function applySearch(q) {
  const lower = (q || '').toLowerCase();
  filtered = lower ? entries.filter(e => (e.filename || e.original_filename || '').toLowerCase().includes(lower)) : [...entries];
  renderList();
}

function renderList() {
  const el = document.getElementById('refLibList');
  if (!el) return;
  if (!filtered.length) {
    el.innerHTML = '<div class="lgjs-centered-empty">Sin resultados</div>';
    return;
  }
  el.innerHTML = filtered.map(entry => {
    const name = entry.filename || entry.original_filename || 'unknown';
    const dur = entry.duration_sec != null ? `${Math.floor(entry.duration_sec / 60)}:${String(Math.floor(entry.duration_sec % 60)).padStart(2, '0')}` : '';
    const lufs = entry.lufs != null ? `${entry.lufs.toFixed(1)} LUFS` : '';
    const peak = entry.peak_db != null ? `${entry.peak_db.toFixed(1)} dB` : '';
    const sr = entry.sr ? `${(entry.sr / 1000).toFixed(1)}kHz` : '';
    return `<div class="ref-lib-row ${selected?.id === entry.id ? 'ref-lib-row--selected' : ''}" data-id="${entry.id}">
      <span class="ref-lib-name">${name}</span>
      <span class="ref-lib-meta"><span>${dur}</span><span>${lufs}</span><span>${peak}</span><span>${sr}</span></span>
    </div>`;
  }).join('');
  el.querySelectorAll('.ref-lib-row').forEach(row => {
    row.addEventListener('click', () => {
      const entry = entries.find(e => e.id === row.dataset.id);
      if (entry) selectEntry(entry);
    });
  });
}

function selectEntry(entry) {
  selected = entry;
  state.set('reference', { ...(state.get('reference') || {}), libraryId: entry.id, file: null });
  const label = document.getElementById('ref-file-label');
  if (label) label.textContent = entry.filename || entry.original_filename || entry.id;
  window.dispatchEvent(new CustomEvent('lgmdm:ref-selected', { detail: { entry } }));
  closeModal();
}

function closeModal() {
  const modal = document.getElementById('refLibModal');
  if (modal) modal.style.display = 'none';
}

export async function reload({ retryOnAuth = false } = {}) {
  const status = document.getElementById('refLibStatus');
  const list = document.getElementById('refLibList');
  if (status) status.textContent = 'Cargando…';
  if (list) list.innerHTML = '';
  try {
    const res = await apiFetch(`${apiBase()}/reference-library`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    entries = data.entries || data.library || data || [];
    if (!Array.isArray(entries)) entries = [];
    filtered = [...entries];
    if (status) status.textContent = `${entries.length} referencias disponibles`;
    renderList();
  } catch (err) {
    if (status) status.textContent = `Error: ${err.message}`;
    if (retryOnAuth && !getToken()) {
      window.addEventListener('lgmdm:authenticated', () => reload(), { once: true });
    }
  }
}

export function open() {
  buildModal();
  const modal = document.getElementById('refLibModal');
  if (modal) modal.style.display = '';
  const search = document.getElementById('refLibSearch');
  if (search) { search.value = ''; search.focus(); }
  filtered = [...entries];
  renderList();
  if (!entries.length) reload();
}

export function init() {
  if (init.done) return;
  init.done = true;
  const trigger = document.getElementById('btnOpenRefLib') || document.getElementById('refFileInput')?.closest('.param')?.querySelector('button');
  trigger?.addEventListener('click', open);
  window.addEventListener('lgmdm:authenticated', () => reload({ retryOnAuth: true }));
}
