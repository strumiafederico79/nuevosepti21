// js/preview-controller.js — Server-side preview completo (reemplaza preview.js)
import { apiFetch, apiBase } from './api.js';
import { collectParams } from './params.js';
import * as state from './state.js';
import { getChainOverrides } from './master-console.js';

const DEBOUNCE_MS = 1500;
const DEFAULT_DURATION = 25;

let running = false;
let activePromise = null;
let renderSession = null;
let sourceSession = null;
let sessionSeq = 0;
let sourceSeq = 0;
let requestTimer = null;
let ready = false;
let previewUrl = null;
let previewSourceId = null;
let previewSourceMeta = null;
let wired = false;

// GR live curves
let liveCurves = null;
let liveRafId = null;

// Cached top-level metrics + spectrum from fetchAndPublishMeters
let cachedTopMetrics = null;
let cachedSpectrum = null;
let progressTimer = null;
let previewProgress = 0;

function emit(name, detail) {
  window.dispatchEvent(new CustomEvent(`lgmdm:${name}`, { detail }));
}

function dispatchMetrics(metrics) {
  window.dispatchEvent(new CustomEvent('lgmdm:metrics', { detail: { metrics } }));
}

function setState(s, text, progress = null) {
  updateRenderStatus(s, text, progress);
  emit('preview-state', { state: s, text, progress });
}

function updateRenderStatus(state, text, progress = null) {
  const status = document.getElementById('previewRenderStatus');
  const label = document.getElementById('previewRenderText');
  const percent = document.getElementById('previewRenderPercent');
  const track = status?.querySelector('[role="progressbar"]');
  const fill = document.getElementById('previewRenderFill');
  if (!status || !label || !percent || !track || !fill) return;

  const inactive = state === 'disabled' || state === 'waiting';
  status.hidden = inactive;
  const value = Number.isFinite(Number(progress)) ? Math.max(0, Math.min(100, Number(progress))) : previewProgress;
  previewProgress = value;
  label.textContent = text;
  percent.textContent = `${Math.round(value)}%`;
  fill.style.width = `${value}%`;
  track.setAttribute('aria-valuenow', String(Math.round(value)));
}

function stopRenderProgress() {
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = null;
}

function startRenderProgress(initial = 8) {
  stopRenderProgress();
  previewProgress = initial;
  progressTimer = setInterval(() => {
    previewProgress = Math.min(92, previewProgress + Math.max(1, (92 - previewProgress) * 0.12));
    updateRenderStatus('processing', 'Renderizando preview…', previewProgress);
  }, 600);
}

function isEnabled() {
  return document.getElementById('s-livepreview')?.checked === true;
}

function getDuration() { return DEFAULT_DURATION; }

function clearPreviewAudio() {
  const wrap = document.getElementById('previewAudioWrap');
  if (wrap) {
    wrap.querySelectorAll('audio').forEach(a => { try { a.pause(); } catch {} });
    wrap.replaceChildren();
  }
  if (previewUrl) { try { URL.revokeObjectURL(previewUrl); } catch {} previewUrl = null; }
  ready = false;
  const pb = document.getElementById('previewPlayBtn');
  const sp = document.getElementById('previewStopBtn');
  if (pb) pb.disabled = true;
  if (sp) sp.disabled = true;
  emit('preview-ready', { ready: false });
}

function renderAudio(blob) {
  if (!(blob instanceof Blob) || blob.size === 0) throw new Error('Preview vacío');
  const wrap = document.getElementById('previewAudioWrap');
  if (!wrap) throw new Error('DOM roto: #previewAudioWrap');
  clearPreviewAudio();
  previewUrl = URL.createObjectURL(blob);
  const audio = document.createElement('audio');
  audio.controls = true;
  audio.preload = 'metadata';
  audio.src = previewUrl;
  audio.dataset.previewReady = 'true';
  wrap.appendChild(audio);
  ready = true;
  const pb = document.getElementById('previewPlayBtn');
  const sp = document.getElementById('previewStopBtn');
  if (pb) pb.disabled = false;
  if (sp) sp.disabled = false;
  emit('preview-ready', { ready: true, audio });
}

function cancelSource() {
  if (sourceSession?.controller) try { sourceSession.controller.abort(); } catch {}
  sourceSession = null;
}

function clearSourceSnapshot() {
  previewSourceId = null;
  previewSourceMeta = null;
  emit('preview-source-state', { state: 'empty', sourceId: null, meta: null });
}

async function createOriginalSnapshot() {
  if (!isEnabled()) return false;
  const file = state.get('selectedFile');
  if (!file) return false;
  if (previewSourceId) return true;
  if (sourceSession?.promise) return sourceSession.promise;

  cancelSource();
  const current = { id: ++sourceSeq, cancelled: false, controller: new AbortController(), promise: null };
  sourceSession = current;
  setState('source-processing', `Preparando snapshot de ${getDuration()} s…`, 0);
  current.promise = (async () => {
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('duration_sec', String(getDuration()));
      fd.append('output_format', 'wav');
      fd.append('output_bit_depth', '24');
      const res = await apiFetch(`${apiBase()}/preview/source`, {
        method: 'POST', body: fd, signal: current.controller.signal, timeout: 120000, maxRetries: 0,
      });
      if (sourceSession !== current || current.cancelled) return false;
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data?.source_id) throw new Error('Sin source_id');
      previewSourceId = data.source_id;
      previewSourceMeta = { duration_sec: data.duration_sec ?? getDuration(), source_sha256: data.source_sha256 || null };
      emit('preview-source-state', { state: 'ready', sourceId: previewSourceId, meta: previewSourceMeta });
      return true;
    } catch (err) {
      if (err?.name === 'AbortError') return false;
      clearSourceSnapshot();
      setState('error', `Error snapshot: ${err.message}`);
      return false;
    } finally {
      if (sourceSession === current) sourceSession = null;
    }
  })();
  return current.promise;
}

async function renderPreview() {
  if (!previewSourceId) throw new Error('Sin snapshot');
  const params = { ...collectParams(), ...getChainOverrides() };
  startRenderProgress(8);
  setState('processing', 'Renderizando preview…', 8);
  const res = await apiFetch(`${apiBase()}/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json;charset=UTF-8' },
    body: JSON.stringify({ preview_source_id: previewSourceId, preview_duration_sec: getDuration(), params }),
    timeout: 120000, maxRetries: 0,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  renderAudio(await res.blob());
  stopRenderProgress();
  setState('ready', `Preview de ${getDuration()} s listo`, 100);
  fetchAndPublishMeters(previewSourceId);
  return true;
}

async function fetchAndPublishMeters(sourceId) {
  try {
    const res = await apiFetch(`${apiBase()}/preview/meters/${sourceId}`, {
      method: 'GET', timeout: 15000, maxRetries: 0,
    });
    if (!res.ok) return;
    const payload = await res.json();
    const cm = payload?.chain_meters || payload?.chainMeters || payload;
    dispatchMetrics({
      chain_meters: cm,
      peak_db: cm?.post_limiter?.peak_db,
      rms_db: cm?.post_limiter?.rms_db,
      lufs_momentary: cm?.post_limiter?.lufs,
      true_peak_db: cm?.post_limiter?.peak_db,
      stereo_correlation: cm?.post_limiter?.stereo_correlation,
      mono_compatibility_db: null,
      comp_meters: cm?.comp || {},
      limiter_meters: cm?.limiter || {},
      glue_meters: cm?.glue || {},
      mb_meters: cm?.mb || {},
      spectrum: cm?.spectrum || null,
    });
    cachedTopMetrics = {
      peak_db: cm?.post_limiter?.peak_db,
      rms_db: cm?.post_limiter?.rms_db,
      lufs_momentary: cm?.post_limiter?.lufs,
      true_peak_db: cm?.post_limiter?.peak_db,
      stereo_correlation: cm?.post_limiter?.stereo_correlation,
      mono_compatibility_db: null,
    };
    cachedSpectrum = cm?.spectrum || null;
    liveCurves = {
      comp: cm?.comp?.curve || [], compHopMs: cm?.comp?.curve_hop_ms || 0,
      glue: cm?.glue?.curve || [], glueHopMs: cm?.glue?.curve_hop_ms || 0,
      low: cm?.mb?.low_curve || [], mid: cm?.mb?.mid_curve || [], high: cm?.mb?.high_curve || [],
      mbHopMs: cm?.mb?.curve_hop_ms || 0,
    };
  } catch {}
}

function curveValueAt(curve, hopMs, t) {
  if (!Array.isArray(curve) || !curve.length || !hopMs) return 0;
  return curve[Math.min(curve.length - 1, Math.max(0, Math.round((t * 1000) / hopMs)))] ?? 0;
}

function liveGRTick(audio) {
  if (!audio || audio.paused || audio.ended) { liveRafId = null; return; }
  if (liveCurves) {
    const t = audio.currentTime;
    dispatchMetrics({
      ...cachedTopMetrics,
      spectrum: cachedSpectrum,
      comp_meters: { gr_db: curveValueAt(liveCurves.comp, liveCurves.compHopMs, t) },
      glue_meters: { gr_db: curveValueAt(liveCurves.glue, liveCurves.glueHopMs, t) },
      mb_meters: {
        low_gr_db: curveValueAt(liveCurves.low, liveCurves.mbHopMs, t),
        mid_gr_db: curveValueAt(liveCurves.mid, liveCurves.mbHopMs, t),
        high_gr_db: curveValueAt(liveCurves.high, liveCurves.mbHopMs, t),
      },
    });
  }
  liveRafId = requestAnimationFrame(() => liveGRTick(audio));
}

function stopLiveGR() {
  if (liveRafId != null) { cancelAnimationFrame(liveRafId); liveRafId = null; }
}

function cancelRender() {
  if (renderSession?.controller) try { renderSession.controller.abort(); } catch {}
  renderSession = null;
  running = false;
  activePromise = null;
  stopRenderProgress();
}

function scheduleRender() {
  clearTimeout(requestTimer);
  if (!isEnabled() || !state.get('selectedFile')) return;
  requestTimer = setTimeout(() => { start().catch(() => {}); }, DEBOUNCE_MS);
  setState('waiting', `Esperando ${DEBOUNCE_MS / 1000} s sin cambios…`);
}

export async function start() {
  if (!isEnabled()) { setState('disabled', 'Preview deshabilitado'); return false; }
  if (!state.get('selectedFile')) { setState('error', 'Cargá un archivo'); return false; }
  if (running && activePromise) return activePromise;

  const sourceReady = await createOriginalSnapshot();
  if (!sourceReady || !previewSourceId) return false;
  clearPreviewAudio();

  const current = { id: ++sessionSeq, cancelled: false, controller: new AbortController(), sourceId: previewSourceId };
  renderSession = current;
  running = true;
  setState('processing', `Procesando Preview de ${getDuration()} s…`, 0);

  activePromise = renderPreview()
    .catch(err => { if (!current.cancelled) { clearPreviewAudio(); stopRenderProgress(); setState('error', `Error: ${err.message}`, 0); } return false; })
    .finally(() => { if (renderSession === current) { renderSession = null; running = false; activePromise = null; } });
  return activePromise;
}

export function stop() {
  clearTimeout(requestTimer); requestTimer = null;
  cancelRender(); clearPreviewAudio();
  previewProgress = 0;
  setState('disabled', 'Preview detenido', 0);
}

export { scheduleRender as request, isEnabled, clearSourceSnapshot as reset };

export function init() {
  if (init.done) return;
  init.done = true;

  const toggle = document.getElementById('s-livepreview');
  const pb = document.getElementById('previewPlayBtn');
  const sp = document.getElementById('previewStopBtn');
  const getAudio = () => document.getElementById('previewAudioWrap')?.querySelector('audio');

  toggle?.addEventListener('change', () => {
    if (toggle.checked) scheduleRender();
    else stop();
  });

  pb?.addEventListener('click', () => getAudio()?.play().catch(() => {}));
  sp?.addEventListener('click', () => { const a = getAudio(); if (a) { a.pause(); a.currentTime = 0; } });

  window.addEventListener('lgmdm:file-selected', () => {
    cancelRender(); cancelSource(); clearPreviewAudio(); clearSourceSnapshot();
    if (isEnabled()) scheduleRender();
  });

  window.addEventListener('lgmdm:param-change', () => {
    if (!isEnabled()) return;
    cancelRender(); clearPreviewAudio(); scheduleRender();
  });

  window.addEventListener('lgmdm:preview-ready', (e) => {
    const audio = e.detail?.audio;
    if (!audio) return;
    audio.addEventListener('play', () => { if (liveRafId == null) liveGRTick(audio); });
    audio.addEventListener('pause', stopLiveGR);
    audio.addEventListener('ended', stopLiveGR);
  });
}
