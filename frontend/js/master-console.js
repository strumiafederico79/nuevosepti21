// js/master-console.js — Mirror controls, A/B, VU meters, animated waveform
import { cachedEl } from './00-dom-safety.js';
import * as state from './state.js';
import * as studio from './studio-controller.js';
import { formatDuration } from './file.js';

const _console = {
  raf: 0, start: performance.now(), playing: false, audio: null,
  ab: 'master', applying: false,
  metrics: null, waveHistory: [],
};

const refs = {
  input: ['s-ingain', 'consoleInputFader'],
  compThreshold: ['s-thresh', 'consoleCompThreshold'],
  compRatio: ['s-ratio', 'consoleCompRatio'],
  stereo: ['s-width', 'consoleStereoFader'],
  limiter: ['s-ceiling', 'consoleLimiterFader'],
};

const _mirrored = new WeakSet();

function mirror(srcId, dstId, converters) {
  const src = cachedEl(srcId), dst = cachedEl(dstId);
  if (!src || !dst) return;
  if (_mirrored.has(src) || _mirrored.has(dst)) return;
  _mirrored.add(src); _mirrored.add(dst);
  const toUI = converters?.toUI ?? (v => v);
  const fromUI = converters?.fromUI ?? (v => v);
  dst.value = toUI(src.value);
  const event = dst.tagName === 'SELECT' || dst.type === 'checkbox' ? 'change' : 'input';
  dst.addEventListener(event, () => {
    src.value = fromUI(dst.value);
    src.dispatchEvent(new Event(event, { bubbles: true }));
    updateReadouts(); updateStageCards();
  });
  src.addEventListener('input', () => { dst.value = toUI(src.value); updateReadouts(); });
  src.addEventListener('change', () => { dst.value = toUI(src.value); updateReadouts(); updateStageCards(); });
}

export function formatDb(v) { return `${v >= 0 ? '+' : ''}${Number(v).toFixed(1)} dB`; }
export function ceilingDb(v) { return 20 * Math.log10(Math.max(0.01, Number(v))); }

export function updateReadouts() {
  const input = Number(cachedEl('s-ingain')?.value ?? 0);
  const ct = Number(cachedEl('s-thresh')?.value ?? -18);
  const cr = Number(cachedEl('s-ratio')?.value ?? 4);
  const sw = Number(cachedEl('s-width')?.value ?? 1);
  const ceil = Number(cachedEl('s-ceiling')?.value ?? -1.0);
  if (cachedEl('consoleInputReadout')) cachedEl('consoleInputReadout').textContent = formatDb(input);
  if (cachedEl('consoleCompReadout')) cachedEl('consoleCompReadout').textContent = `${ct.toFixed(1)} dB · ${cr.toFixed(1)}:1`;
  if (cachedEl('consoleStereoReadout')) cachedEl('consoleStereoReadout').textContent = `${Math.round(sw * 100)}%`;
  if (cachedEl('consoleLimiterControlReadout')) cachedEl('consoleLimiterControlReadout').textContent = `${Number(ceil).toFixed(1)} dB`;
  if (cachedEl('consoleInputGr')) cachedEl('consoleInputGr').textContent = formatDb(input);
  if (cachedEl('consoleStereoGr')) cachedEl('consoleStereoGr').textContent = `WIDTH ${Math.round(sw * 100)}%`;
  if (cachedEl('consoleLimiterReadout')) cachedEl('consoleLimiterReadout').textContent = `CEILING ${Number(ceil).toFixed(1)}`;
  if (cachedEl('consoleCompGr')) cachedEl('consoleCompGr').textContent = `GR 0.0 dB`;
}

export function updateStageCards() {
  const bypass = studio.getBypassState();
  document.querySelectorAll('.lg-stage-card').forEach(card => {
    const stage = card.dataset.stage;
    card.classList.toggle('bypassed', !!bypass[stage]);
    const em = card.querySelector('em');
    if (em) em.textContent = bypass[stage] ? 'BYPASS' : 'ACTIVE';
  });
}

export function setAB(mode) {
  _console.ab = mode;
  cachedEl('consoleABReadout')?.replaceChildren(document.createTextNode(mode === 'master' ? 'MASTER' : 'ORIGINAL'));
  cachedEl('consoleABMaster')?.classList.toggle('active', mode === 'master');
  cachedEl('consoleABOriginal')?.classList.toggle('active', mode === 'original');
  const abBtn = cachedEl('consoleABToggle');
  if (abBtn) abBtn.setAttribute('aria-pressed', mode === 'master' ? 'true' : 'false');
  const audio = getPreviewAudio();
  if (audio) audio.dataset.abMode = mode;
}

export function toggleAB() { setAB(_console.ab === 'master' ? 'original' : 'master'); }

export function getPreviewAudio() {
  return document.querySelector('#previewAudioWrap audio, #mxrServerPreviewAudio');
}

function clamp01(v) { return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0)); }
function metricAmp(db, floor = -72) { return clamp01((Number(db ?? floor) - floor) / (0 - floor)); }

export function drawWaveform() {
  const canvas = cachedEl('lgmdmWaveformCanvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect(); const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(320, Math.floor(rect.width * dpr)), h = Math.max(120, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(116,230,255,.09)'; ctx.lineWidth = 1;
  for (let i = 1; i < 8; i++) { const y = h / 8 * i; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  const m = _console.metrics || {};
  const peakAmp = metricAmp(m.peak_db, -72);
  const rmsAmp = metricAmp(m.rms_db, -72);
  _console.waveHistory.push({ peak: peakAmp, rms: rmsAmp });
  if (_console.waveHistory.length > 90) _console.waveHistory.shift();
  const hist = _console.waveHistory;
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, 'rgba(87,230,255,.25)'); grad.addColorStop(.5, 'rgba(169,140,255,.85)'); grad.addColorStop(1, 'rgba(87,230,255,.25)');
  const mid = h / 2;
  ctx.strokeStyle = grad; ctx.lineWidth = Math.max(1, 1.4 * dpr);
  ctx.beginPath();
  const phase = (performance.now() - _console.start) / 500;
  for (let i = 0; i < 220; i++) {
    const t = i / 219, idx = Math.min(hist.length - 1, Math.floor(t * (hist.length - 1)));
    const item = hist[idx] || { peak: peakAmp, rms: rmsAmp };
    const env = Math.max(.03, item.rms * .75 + item.peak * .25);
    const texture = .45 * Math.sin(t * 34 + phase) + .2 * Math.sin(t * 87 - phase * .6) + .12 * Math.sin(t * 13 + phase * .3);
    const y = mid - texture * env * h * .38;
    i ? ctx.lineTo(t * w, y) : ctx.moveTo(t * w, y);
  }
  ctx.stroke();
  ctx.beginPath();
  hist.forEach((item, i) => {
    const x = hist.length === 1 ? 0 : i / (hist.length - 1) * w;
    const y = mid - item.rms * h * .36;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = 'rgba(255,202,101,.9)'; ctx.lineWidth = Math.max(1, 1 * dpr); ctx.stroke();
}

export function updateConsoleStereoVu() {
  const l = cachedEl('consoleMeterL'), r = cachedEl('consoleMeterR');
  if (!l || !r) return;
  const m = _console.metrics || {};
  const peak = metricAmp(m.peak_db, -60);
  const corr = Math.max(-1, Math.min(1, Number(m.stereo_correlation ?? 1)));
  const spread = (1 - Math.max(0, corr)) * 0.18;
  l.style.height = `${Math.max(3, Math.min(100, (peak * (1 + spread)) * 100))}%`;
  r.style.height = `${Math.max(3, Math.min(100, (peak * (1 - spread)) * 100))}%`;
}

export function setStatus(text, active = false) {
  cachedEl('consoleStatus')?.replaceChildren(document.createTextNode(text));
  document.querySelector('.lg-status-dot')?.classList.toggle('active', active);
}

export function syncTrackInfo() {
  const file = state.get('selectedFile');
  if (!file) {
    cachedEl('consoleTrackTitle')?.replaceChildren(document.createTextNode('Sin archivo cargado'));
    cachedEl('consoleTrackMeta')?.replaceChildren(document.createTextNode('Esperando señal'));
    return;
  }
  cachedEl('consoleTrackTitle')?.replaceChildren(document.createTextNode(file.name.replace(/\.[^/.]+$/, '')));
  cachedEl('consoleTrackMeta')?.replaceChildren(document.createTextNode(`${file.type || 'audio'} · ${(file.size / 1024 / 1024).toFixed(1)} MB`));
  setStatus('Audio cargado · listo para analizar', true);
}

export function syncMetersFromDom() {
  const map = [['meterPeakReadout', 'consolePeak'], ['meterLufsReadout', 'consoleLufs'], ['meterTruePeakReadout', 'consoleTruePeak'], ['meterRmsReadout', 'consoleRms'], ['stereoMeterReadout', 'consoleCorr']];
  for (const [src, dst] of map) { const a = cachedEl(src), b = cachedEl(dst); if (a && b && a.textContent) b.textContent = a.textContent.replace(/^corr:\s*/i, ''); }
  updateConsoleStereoVu();
}

export function syncChainMeters(metrics) {
  if (!metrics) return;
  const chain = metrics.chain_meters || metrics.chainMeters || {};
  const comp = chain.comp || metrics.comp_meters || {};
  const glue = chain.glue || metrics.glue_meters || {};
  const limiter = chain.limiter || metrics.limiter_meters || {};
  const compGr = Number(comp.gr_db ?? metrics.comp_gr_db ?? 0);
  const glueGr = Number(glue.gr_db ?? 0);
  const limGr = Number(limiter.gr_db ?? metrics.limiter_gr_db ?? 0);
  if (cachedEl('consoleCompGr')) cachedEl('consoleCompGr').textContent = `GR ${(Number.isFinite(compGr) ? compGr : 0).toFixed(1)} dB`;
  if (cachedEl('consoleLimiterGr')) cachedEl('consoleLimiterGr').textContent = `GR ${(Number.isFinite(limGr) ? limGr : 0).toFixed(1)} dB`;
  if (cachedEl('consoleGlueGr')) cachedEl('consoleGlueGr').textContent = `GR ${(Number.isFinite(glueGr) ? glueGr : 0).toFixed(1)} dB`;
  if (cachedEl('consoleOutputReadout')) cachedEl('consoleOutputReadout').textContent = metrics.output_lufs != null ? `${Number(metrics.output_lufs).toFixed(1)} LUFS` : (cachedEl('consoleLufs')?.textContent || '-∞ LUFS');
}

export function getChainOverrides() {
  // Fuente única de verdad por plugin: studio.isPluginBypassed(key) ya sabe
  // resolver el mecanismo correcto para cada uno (stageBypass / bypassInput /
  // virtualBypass) y usa las keys reales de PLUGINS — evita el bug de
  // nombres cruzados (activeSet.has('comp') vs key real 'compressor', etc.)
  // que dejaba varios plugins bypasseados sin importar lo que hiciera el
  // usuario en el rack.
  const isBypassed = (key) => (studio.isPluginBypassed ? studio.isPluginBypassed(key) : true);
  return {
    comp_bypass: isBypassed('compressor'),
    stereo_bypass: isBypassed('stereo'),
    mb_bypass: isBypassed('multiband'),
    glue_bypass: isBypassed('glue'),
    ms_comp_bypass: isBypassed('ms_comp'),
    mb_stereo_bypass: isBypassed('mb_stereo'),
    clipper_bypass: isBypassed('clipper'),
    dyneq_bypass: isBypassed('dynamic_eq'),
    // Sin card/checkbox en el rack todavía — quedan forzados bypass=true
    // hasta que se les arme UI (no son "checkboxes rotos", no existen).
    reso_bypass: true,
    tonal_balance_bypass: true,
    lp_bypass: true,
    ms_eq_bypass: true,
    // Política conservadora: el Limiter NUNCA bypass (brick-wall final).
    // Sep 11: snapshots con lbypass=True mostraban peaks hasta +17 dBFS →
    // clipping digital cuantizando a PCM_16.
    limiter_bypass: false,
  };
}

let wired = false;
export function wire() {
  if (wired) return;
  wired = true;
  mirror(...refs.input); mirror(...refs.compThreshold); mirror(...refs.compRatio); mirror(...refs.stereo); mirror(...refs.limiter);
  cachedEl('consoleABMaster')?.addEventListener('click', () => setAB('master'));
  cachedEl('consoleABOriginal')?.addEventListener('click', () => setAB('original'));
  cachedEl('consoleABToggle')?.addEventListener('click', toggleAB);
  document.querySelectorAll('.lg-stage-card').forEach(btn => btn.addEventListener('click', () => {
    const stage = btn.dataset.stage;
    const pluginKey = { comp: 'compressor', stereo: 'stereo', limiter: 'limiter' }[stage];
    if (pluginKey) {
      const isBypassed = studio.isStageBypassed(stage);
      studio.setPluginBypass(pluginKey, !isBypassed);
      updateStageCards();
      updateReadouts();
    }
  }));
  state.subscribe('selectedFile', syncTrackInfo);
  syncTrackInfo(); updateReadouts(); updateStageCards();
  const TARGET_FPS = 30;
  const FRAME_MS = 1000 / TARGET_FPS;
  let lastFrame = 0;
  const tick = (now) => {
    const onConsole = document.body.dataset.workspace === 'console';
    if (onConsole && (now - lastFrame) >= FRAME_MS) {
      drawWaveform(); syncMetersFromDom(); lastFrame = now;
    }
    _console.audio = getPreviewAudio();
    const audio = _console.audio;
    if (audio && onConsole) {
      cachedEl('consoleTime').textContent = formatDuration(audio.currentTime);
      cachedEl('consoleDuration').textContent = formatDuration(audio.duration);
      const ph = cachedEl('consolePlayhead');
      if (Number.isFinite(audio.duration) && audio.duration > 0 && ph) ph.style.left = `${audio.currentTime / audio.duration * 100}%`;
    }
    _console.raf = requestAnimationFrame(tick);
  };
  _console.raf = requestAnimationFrame(tick);
  window.addEventListener('lgmdm:metrics', e => { _console.metrics = e.detail?.metrics || null; syncChainMeters(_console.metrics); });
}
