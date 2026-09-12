// js/analysis-view.js — Server analysis renderer (reemplaza analysis.js)
import { apiFetch, apiBase } from './api.js';
import * as state from './state.js';
import { renderAnalysisSingle, renderFFT } from './visualizers.js';

let requestSeq = 0;
let lastData = null;

function emit(name, detail) {
  window.dispatchEvent(new CustomEvent(`lgmdm:${name}`, { detail }));
}

function setStatus(s, text, progress = null) {
  const el = document.getElementById('analysisStatus');
  if (!el) return;
  el.textContent = text;
  el.className = 'analysis-status ' + s;
}

export async function requestAnalysis(options = {}) {
  const file = state.get('selectedFile');
  if (!file) throw new Error('Sin archivo seleccionado');
  const seq = ++requestSeq;
  setStatus('processing', 'Analizando…', 0);
  emit('analysis-state', { state: 'processing', text: 'Analizando…', progress: 0 });
  const fd = new FormData();
  fd.append('file', file, file.name);
  const res = await apiFetch(`${apiBase()}/analysis`, {
    method: 'POST', body: fd, timeout: 30000, maxRetries: 0,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (seq !== requestSeq) return null;
  lastData = data;
  state.set('lastAnalysis', data);
  renderServerAnalysis(data);
  setStatus('ready', 'Análisis completo');
  emit('analysis-state', { state: 'ready', text: 'Análisis completo' });
  emit('analysis-updated', data);
  const sp = Array.isArray(data?.spectrum) ? data.spectrum
    : (data?.bands_db || data?.magnitudes_db || data?.values_db || null);
  if (sp != null || data?.peak_db != null || data?.rms_db != null) {
    window.dispatchEvent(new CustomEvent('lgmdm:metrics', { detail: { metrics: {
      peak_db: data.peak_db, rms_db: data.rms_db,
      lufs_momentary: data.lufs_momentary ?? data.lufs,
      true_peak_db: data.true_peak_db,
      stereo_correlation: data.stereo_correlation,
      chain_meters: data.chain_meters,
      spectrum: sp,
    } } }));
  }
  return data;
}

function renderServerAnalysis(data) {
  const container = document.getElementById('analysisDynamicContent');
  if (!container) return;
  container.innerHTML = '';
  renderAnalysisSingle(data, container);
}

export function update(data) {
  lastData = data;
  renderServerAnalysis(data);
}

export function clear() {
  requestSeq++;
  lastData = null;
  const container = document.getElementById('analysisDynamicContent');
  if (container) container.innerHTML = '';
  setStatus('', 'Idle');
}

export function redraw() {
  if (lastData) renderServerAnalysis(lastData);
}

export function init() {
  if (init.done) return;
  init.done = true;
  window.addEventListener('lgmdm:file-selected', () => clear());
}
