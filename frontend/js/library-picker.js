// js/library-picker.js — modal overlay para elegir tracks de la librería del server.
// Crea el DOM lazy al primer open. Soporta búsqueda client-side, upload nuevo,
// seleccionar y borrar. El fetch a /library/{id}/download retorna blob que envolvemos
// en File object para mantener compatibilidad con state.set('selectedFile', file) y
// el flujo existente de submitJob()/preview.

import { apiFetch, apiBase } from './api.js';
import { showToast } from './00-error-handler.js';
import { formatDuration } from './file.js';

let modalEl = null;
let cachedFiles = [];

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function buildModal() {
  const wrap = document.createElement('div');
  wrap.id = 'library-modal';
  wrap.className = 'lp-overlay hidden';
  wrap.innerHTML = `
    <div class="lp-window" role="dialog" aria-label="Librería del servidor">
      <header class="lp-header">
        <h2>📚 Librería</h2>
        <button type="button" class="lp-close" aria-label="Cerrar">×</button>
      </header>
      <div class="lp-toolbar">
        <input id="lp-search" placeholder="Buscar por nombre…" autocomplete="off">
        <label class="lp-upload-btn">
          <span>↑ Subir archivo</span>
          <input type="file" accept="audio/*" hidden>
        </label>
      </div>
      <div class="lp-table-wrap">
        <table class="lp-table">
          <thead><tr>
            <th style="width:48px"></th>
            <th>Nombre</th>
            <th style="width:90px">Duración</th>
            <th style="width:80px">Tamaño</th>
            <th style="width:110px">Subido</th>
            <th style="width:48px"></th>
          </tr></thead>
          <tbody id="lp-rows"><tr><td colspan="6" class="lp-empty">Cargando…</td></tr></tbody>
        </table>
      </div>
      <footer id="lp-status" class="lp-footer">—</footer>
    </div>`;

  wrap.querySelector('.lp-close').addEventListener('click', () => closeLibrary());
  wrap.addEventListener('click', (e) => { if (e.target === wrap) closeLibrary(); });
  wrap.querySelector('#lp-search').addEventListener('input', renderRows);
  wrap.querySelector('.lp-upload-btn input').addEventListener('change', handleUpload);

  document.body.appendChild(wrap);
  return wrap;
}

async function loadFiles() {
  const tbody = modalEl.querySelector('#lp-rows');
  const status = modalEl.querySelector('#lp-status');
  tbody.innerHTML = `<tr><td colspan="6" class="lp-empty">Cargando…</td></tr>`;
  status.textContent = 'Cargando…';
  try {
    const res = await apiFetch(`${apiBase()}/library?limit=200`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    cachedFiles = Array.isArray(data.files) ? data.files : [];
    status.textContent = `${cachedFiles.length} track${cachedFiles.length !== 1 ? 's' : ''} guardado${cachedFiles.length !== 1 ? 's' : ''}`;
  } catch (e) {
    cachedFiles = [];
    status.textContent = 'Error cargando librería';
    showToast(`No pude listar la librería: ${e.message}`, 'error');
  }
  renderRows();
}

function renderRows() {
  if (!modalEl) return;
  const q = (modalEl.querySelector('#lp-search').value || '').toLowerCase().trim();
  const rows = q ? cachedFiles.filter(f => f.original_filename.toLowerCase().includes(q)) : cachedFiles;
  const tbody = modalEl.querySelector('#lp-rows');
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="lp-empty">${q ? `Sin coincidencias para "${escapeHtml(q)}"` : 'La librería está vacía.'}</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(f => `
    <tr data-id="${escapeHtml(f.id)}">
      <td><button type="button" class="lp-play" title="Cargar este track">▶</button></td>
      <td class="lp-name">${escapeHtml(f.original_filename)}</td>
      <td>${formatDuration(f.duration_sec)}</td>
      <td>${(Number(f.size_bytes)/1048576).toFixed(1)} MB</td>
      <td>${f.created_at ? new Date(f.created_at * 1000).toLocaleDateString() : '—'}</td>
      <td><button type="button" class="lp-del" title="Borrar">✕</button></td>
    </tr>`).join('');
  tbody.querySelectorAll('.lp-play').forEach(b =>
    b.addEventListener('click', () => selectFromLibrary(b.closest('tr').dataset.id)));
  tbody.querySelectorAll('.lp-del').forEach(b =>
    b.addEventListener('click', (ev) => { ev.stopPropagation(); deleteFromLibrary(b.closest('tr').dataset.id); }));
  // Doble click en la fila también selecciona.
  tbody.querySelectorAll('tr[data-id]').forEach(tr =>
    tr.addEventListener('dblclick', () => selectFromLibrary(tr.dataset.id)));
}

async function selectFromLibrary(fileId) {
  const meta = cachedFiles.find(f => f.id === fileId);
  if (!meta) return;
  try {
    const res = await apiFetch(`${apiBase()}/library/${fileId}/download`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const file = new File([blob], meta.original_filename, { type: blob.type || 'audio/wav' });
    window.dispatchEvent(new CustomEvent('lgmdm:file-selected', { detail: { file } }));
    closeLibrary();
    showToast(`Track cargado: ${meta.original_filename}`, 'success', 3500);
  } catch (e) {
    showToast(`No pude cargar: ${e.message}`, 'error');
  }
}

async function deleteFromLibrary(fileId) {
  const meta = cachedFiles.find(f => f.id === fileId);
  if (!confirm(`¿Borrar "${meta?.original_filename || fileId}" de la librería?\nEsta acción no se puede deshacer.`)) return;
  try {
    const res = await apiFetch(`${apiBase()}/library/${fileId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    cachedFiles = cachedFiles.filter(f => f.id !== fileId);
    showToast('Archivo borrado de la librería', 'success', 2500);
    renderRows();
    const status = modalEl?.querySelector('#lp-status');
    if (status) status.textContent = `${cachedFiles.length} track${cachedFiles.length !== 1 ? 's' : ''} restante${cachedFiles.length !== 1 ? 's' : ''}`;
  } catch (e) {
    showToast(`Error borrando: ${e.message}`, 'error');
  }
}

async function handleUpload(ev) {
  const f = ev.target.files?.[0];
  if (!f) return;
  ev.target.value = '';
  const fd = new FormData();
  fd.append('file', f);
  const status = modalEl?.querySelector('#lp-status');
  if (status) status.textContent = `Subiendo ${f.name}…`;
  try {
    const res = await apiFetch(`${apiBase()}/library/upload`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    showToast('Archivo agregado a la librería', 'success', 2500);
    await loadFiles();
  } catch (e) {
    showToast(`Error subiendo: ${e.message}`, 'error');
    if (status) status.textContent = 'Falló la subida';
  }
}

export async function openLibrary() {
  if (!modalEl) modalEl = buildModal();
  modalEl.classList.remove('hidden');
  await loadFiles();
  // Foco automático en search box tras animación.
  setTimeout(() => modalEl?.querySelector('#lp-search')?.focus(), 120);
}

export function closeLibrary() {
  if (modalEl) modalEl.classList.add('hidden');
}
