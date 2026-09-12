// js/preview-controller.js — Server-side preview completo (reemplaza preview.js)
import { apiFetch, apiBase } from './api.js';
import { collectParams } from './params.js';
import { getChainOverrides } from './master-console.js';
import * as state from './state.js';

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
let _cachedTop = {};

// AnalyserNode local conectado al <audio>: provee spectrum + peak/rms reales
// durante la reproducción sin volver a procesar el WAV server-side.
const BAND_EDGES = [20, 60, 170, 350, 1000, 2500, 6000, 12000, 16000];
let liveSpectrum = {
  audioCtx: null,
  source: null,
  spectrum: null,
  tallies: null,
  freqBuf: null,
  timeBuf: null,
};

function byteToDbBands(byteArr, bandCount) {
  // Reduce FFT bins (size=byteArr.length) a `bandCount` bandas en escala dB
  // alineadas con BAND_EDGES (logarítmicas). Devuelve array plano de bandCount.
  const sr = liveSpectrum.audioCtx?.sampleRate || 48000;
  const nyquist = sr / 2;
  const binsPerBand = [];
  for (let i = 0; i < bandCount; i++) {
    const fLo = BAND_EDGES[i], fHi = BAND_EDGES[i + 1] ?? nyquist;
    const bLo = Math.floor((fLo / nyquist) * byteArr.length);
    const bHi = Math.max(bLo + 1, Math.floor((fHi / nyquist) * byteArr.length));
    let sum = 0;
    for (let k = bLo; k < bHi && k < byteArr.length; k++) sum += (byteArr[k] - 128) / 128;
    const avg = sum / Math.max(1, bHi - bLo);
    binsPerBand.push(20 * Math.log10(Math.abs(avg) || 1e-4));
  }
  return binsPerBand;
}

function ensureAnalyser(audio) {
  // Web Audio API solo permite UN MediaElementSourceNode por elemento HTMLMediaElement.
  // Si ya conectamos este mismo <audio>, reusamos el source y el context.
  if (audio && audio._lgmdm_source && liveSpectrum.audioCtx?.state !== 'closed') {
    liveSpectrum.source = audio._lgmdm_source;
    liveSpectrum.audioCtx = audio._lgmdm_ctx;
    if (liveSpectrum.spectrum && liveSpectrum.freqBuf) return true;
  }
  if (liveSpectrum.source) teardownAnalyser();
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return false;
    const ctx = new AC();
    if (ctx.state === 'suspended') ctx.resume().catch(() => {});
    const src = ctx.createMediaElementSource(audio);
    const spec = ctx.createAnalyser();
    spec.fftSize = 2048;
    spec.smoothingTimeConstant = 0.7;
    const tall = ctx.createAnalyser();
    tall.fftSize = 1024;
    tall.smoothingTimeConstant = 0.3;
    src.connect(spec);
    src.connect(tall);
    // Passthrough: una vez que createMediaElementSource() se ejecuta, el audio deja
    // de salir por defecto y solo suena si lo ruteamos explícitamente al destino.
    // gain=1 garantiza volumen unity sin doble procesamiento ni distorsión.
    const sink = ctx.createGain();
    sink.gain.value = 1;
    src.connect(sink);
    sink.connect(ctx.destination);
    liveSpectrum.audioCtx = ctx;
    liveSpectrum.source = src;
    liveSpectrum.spectrum = spec;
    liveSpectrum.tallies = tall;
    liveSpectrum.freqBuf = new Uint8Array(spec.frequencyBinCount);
    liveSpectrum.timeBuf = new Float32Array(tall.fftSize);
    if (audio) { audio._lgmdm_source = src; audio._lgmdm_ctx = ctx; }
    return true;
  } catch (e) {
    console.warn('[preview] analyser setup failed:', e);
    teardownAnalyser();
    return false;
  }
}

function purgeAnalyserRefs() {
  document.querySelectorAll('#previewAudioWrap audio').forEach(a => {
    a._lgmdm_source = null;
    a._lgmdm_ctx = null;
  });
}

function teardownAnalyser() {
  try { liveSpectrum.source?.disconnect(); } catch {}
  try { liveSpectrum.audioCtx?.close(); } catch {}
  // No purgar audio._lgmdm_* aca — ese cache es por-elemento y puede servir
  // si el browser reusa el mismo <audio>. Solo limpiamos cuando se cambia archivo.
  liveSpectrum = { audioCtx: null, source: null, spectrum: null, tallies: null, freqBuf: null, timeBuf: null };
}

function emit(name, detail) {
  window.dispatchEvent(new CustomEvent(`lgmdm:${name}`, { detail }));
}

function dispatchMetrics(metrics) {
  // BUGFIX (Sep 11): _cachedTop debe actualizarse siempre que lleguen campos
  // top-level (peak/rms/spectrum/chain). Antes solo se refrescaba si había
  // peak_db o chain_meters en la llamada; cuando liveGRTick emitía grPartial
  // sin esos campos, _cachedTop quedaba congelado con los datos del último
  // render → los medidores mostraban números viejos hasta terminar el nuevo.
  const topKeys = ['chain_meters', 'peak_db', 'rms_db', 'lufs_momentary',
                   'true_peak_db', 'stereo_correlation', 'mono_compatibility_db',
                   'spectrum', 'limiter_meters'];
  let dirty = false;
  for (const k of topKeys) {
    if (metrics[k] !== undefined && metrics[k] !== null) { dirty = true; break; }
  }
  if (dirty) {
    _cachedTop = {};
    for (const k of topKeys) {
      if (metrics[k] !== undefined && metrics[k] !== null) _cachedTop[k] = metrics[k];
    }
  }
  window.dispatchEvent(new CustomEvent('lgmdm:metrics', { detail: { metrics: { ..._cachedTop, ...metrics } } }));
}

function setState(s, text, progress = null) {
  emit('preview-state', { state: s, text, progress });
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
      fd.append('output_bit_depth', '16');
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
  setState('processing', 'Renderizando preview…', 0);
  const res = await apiFetch(`${apiBase()}/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json;charset=UTF-8' },
    body: JSON.stringify({ preview_source_id: previewSourceId, preview_duration_sec: getDuration(), params }),
    timeout: 120000, maxRetries: 0,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  renderAudio(await res.blob());
  setState('ready', `Preview de ${getDuration()} s listo`, 100);
  fetchAndPublishMeters(previewSourceId);
  return true;
}

async function fetchAndPublishMeters(sourceId) {
  try {
    const res = await apiFetch(`${apiBase()}/preview/meters/${sourceId}`, {
      method: 'GET', timeout: 15000, maxRetries: 0,
    });
    if (!res.ok) { console.warn('[preview] meters HTTP', res.status); return; }
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
    liveCurves = {
      comp: cm?.comp?.curve || [], compHopMs: cm?.comp?.curve_hop_ms || 0,
      glue: cm?.glue?.curve || [], glueHopMs: cm?.glue?.curve_hop_ms || 0,
      low: cm?.mb?.low_curve || [], mid: cm?.mb?.mid_curve || [], high: cm?.mb?.high_curve || [],
      mbHopMs: cm?.mb?.curve_hop_ms || 0,
    };
  } catch (e) { console.error('[preview] fetchAndPublishMeters failed:', e); }
}

function curveValueAt(curve, hopMs, t) {
  if (!Array.isArray(curve) || !curve.length || !hopMs) return 0;
  return curve[Math.min(curve.length - 1, Math.max(0, Math.round((t * 1000) / hopMs)))] ?? 0;
}

function liveGRTick(audio) {
  if (!audio || audio.paused || audio.ended) { liveRafId = null; return; }
  const t = audio.currentTime;
  // GR curves: interpolación de los snapshots emitidos por el render.
  const grPartial = {};
  if (liveCurves) {
    grPartial.comp_meters = { gr_db: curveValueAt(liveCurves.comp, liveCurves.compHopMs, t) };
    grPartial.glue_meters = { gr_db: curveValueAt(liveCurves.glue, liveCurves.glueHopMs, t) };
    grPartial.mb_meters = {
      low_gr_db: curveValueAt(liveCurves.low, liveCurves.mbHopMs, t),
      mid_gr_db: curveValueAt(liveCurves.mid, liveCurves.mbHopMs, t),
      high_gr_db: curveValueAt(liveCurves.high, liveCurves.mbHopMs, t),
    };
  }
  // AnalyserNode local: spectrum + peak/rms reales del WAV procesado que está sonando.
  const sa = liveSpectrum.spectrum;
  const ta = liveSpectrum.tallies;
  const fb = liveSpectrum.freqBuf;
  const tb = liveSpectrum.timeBuf;
  const partial = {};
  if (sa && fb) {
    sa.getByteFrequencyData(fb);
    partial.spectrum = byteToDbBands(fb, BAND_EDGES.length - 1);
  }
  if (ta && tb) {
    ta.getFloatTimeDomainData(tb);
    let sumSq = 0, peak = 0;
    for (let i = 0; i < tb.length; i++) {
      const v = tb[i];
      sumSq += v * v;
      const a = Math.abs(v);
      if (a > peak) peak = a;
    }
    const rms = Math.sqrt(sumSq / tb.length);
    partial.peak_db = 20 * Math.log10(peak || 1e-9);
    partial.rms_db = 20 * Math.log10(rms || 1e-9);
    partial.live_level = { peak, rms };
  }
  dispatchMetrics({ ...grPartial, ...partial });
  liveRafId = requestAnimationFrame(() => liveGRTick(audio));
}

function stopLiveGR() {
  if (liveRafId != null) { cancelAnimationFrame(liveRafId); liveRafId = null; }
  teardownAnalyser();
}

function cancelRender() {
  if (renderSession?.controller) try { renderSession.controller.abort(); } catch {}
  renderSession = null;
  running = false;
  activePromise = null;
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
    .catch(err => { if (!current.cancelled) { clearPreviewAudio(); setState('error', `Error: ${err.message}`); } return false; })
    .finally(() => { if (renderSession === current) { renderSession = null; running = false; activePromise = null; } });
  return activePromise;
}

export function stop() {
  clearTimeout(requestTimer); requestTimer = null;
  cancelRender(); clearPreviewAudio();
  setState('disabled', 'Preview detenido');
}

export { scheduleRender as request, isEnabled, clearSourceSnapshot as reset };

function updateProgressBar({ state: s, text }) {
  const wrap = document.getElementById('previewProgress');
  const fill = document.getElementById('previewProgressFill');
  const label = document.getElementById('previewProgressText');
  if (!wrap || !fill || !label) return;

  clearTimeout(updateProgressBar._hideTimer);
  wrap.classList.remove('is-done', 'is-error');

  if (s === 'source-processing' || s === 'processing') {
    wrap.hidden = false;
    label.textContent = text || 'Procesando…';
    return;
  }
  if (s === 'ready') {
    wrap.hidden = false;
    wrap.classList.add('is-done');
    label.textContent = text || 'Listo';
    updateProgressBar._hideTimer = setTimeout(() => { wrap.hidden = true; }, 900);
    return;
  }
  if (s === 'error') {
    wrap.hidden = false;
    wrap.classList.add('is-error');
    label.textContent = text || 'Error';
    updateProgressBar._hideTimer = setTimeout(() => { wrap.hidden = true; }, 2500);
    return;
  }
  // 'waiting' (debounce, todavía no hay trabajo real en el server) y
  // 'disabled' no muestran barra — no queremos que el usuario espere una
  // señal de servidor cuando el servidor todavía no está haciendo nada.
  wrap.hidden = true;
}

export function init() {
  if (init.done) return;
  init.done = true;

  window.addEventListener('lgmdm:preview-state', (e) => updateProgressBar(e.detail || {}));

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
    purgeAnalyserRefs();
    cancelRender(); cancelSource(); clearPreviewAudio(); clearSourceSnapshot();
    if (isEnabled()) scheduleRender();
  });

  window.addEventListener('lgmdm:param-change', () => {
    if (!isEnabled()) return;
    // Cancel render en vuelo (si hay) para no desperdiciar CPU; limpiamos también
    // liveCurves y la caché de chain_meters porque sus valores corresponden al audio
    // anterior. La nueva curva llega cuando termine fetchAndPublishMeters post-render.
    cancelRender();
    liveCurves = null;
    _cachedTop = {};
    clearPreviewAudio();
    purgeAnalyserRefs();
    scheduleRender();
  });

  window.addEventListener('lgmdm:preview-ready', (e) => {
    const audio = e.detail?.audio;
    if (!audio) return;
    audio.addEventListener('play', () => {
      ensureAnalyser(audio);
      if (liveRafId == null) liveGRTick(audio);
    });
    audio.addEventListener('seeked', () => ensureAnalyser(audio));
    audio.addEventListener('pause', stopLiveGR);
    audio.addEventListener('ended', stopLiveGR);
  });
}
