// js/master-console.js — Mirror controls, A/B, VU meters
import { cachedEl } from './00-dom-safety.js';
import * as state from './state.js';
import * as studio from './studio-controller.js';
import { formatDuration } from './file.js';

const _console = {
  raf: 0, start: performance.now(), playing: false, audio: null,
  ab: 'master', applying: false,
  metrics: null,
};

const refs = {
  input: ['s-ingain', 'consoleInputFader'],
  compThreshold: ['s-thresh', 'consoleCompThreshold'],
  compRatio: ['s-ratio', 'consoleCompRatio'],
  stereo: ['s-width', 'consoleStereoFader'],
  limiter: ['s-ceiling', 'consoleLimiterFader'],
};

const _mirrored = new WeakSet();

function mirror(srcId, dstId) {
  const src = cachedEl(srcId), dst = cachedEl(dstId);
  if (!src || !dst) return;
  if (_mirrored.has(src) || _mirrored.has(dst)) return;
  _mirrored.add(src); _mirrored.add(dst);
  dst.value = src.value;
  const event = dst.tagName === 'SELECT' || dst.type === 'checkbox' ? 'change' : 'input';
  dst.addEventListener(event, () => {
    src.value = dst.value;
    src.dispatchEvent(new Event(event, { bubbles: true }));
    updateReadouts(); updateStageCards();
  });
  src.addEventListener('input', () => { dst.value = src.value; updateReadouts(); });
  src.addEventListener('change', () => { dst.value = src.value; updateReadouts(); updateStageCards(); });
}

export function formatDb(v) { return `${v >= 0 ? '+' : ''}${Number(v).toFixed(1)} dB`; }
export function ceilingDb(v) { return 20 * Math.log10(Math.max(0.01, Number(v))); }

export function updateReadouts() {
  const input = Number(cachedEl('s-ingain')?.value ?? 0);
  const ct = Number(cachedEl('s-thresh')?.value ?? -18);
  const cr = Number(cachedEl('s-ratio')?.value ?? 4);
  const sw = Number(cachedEl('s-width')?.value ?? 1);
  const ceil = Number(cachedEl('s-ceiling')?.value ?? .891);
  if (cachedEl('consoleInputReadout')) cachedEl('consoleInputReadout').textContent = formatDb(input);
  if (cachedEl('consoleCompReadout')) cachedEl('consoleCompReadout').textContent = `${ct.toFixed(1)} dB · ${cr.toFixed(1)}:1`;
  if (cachedEl('consoleStereoReadout')) cachedEl('consoleStereoReadout').textContent = `${Math.round(sw * 100)}%`;
  if (cachedEl('consoleLimiterControlReadout')) cachedEl('consoleLimiterControlReadout').textContent = `${ceilingDb(ceil).toFixed(1)} dB`;
  if (cachedEl('consoleInputGr')) cachedEl('consoleInputGr').textContent = formatDb(input);
  if (cachedEl('consoleStereoGr')) cachedEl('consoleStereoGr').textContent = `WIDTH ${Math.round(sw * 100)}%`;
  if (cachedEl('consoleLimiterReadout')) cachedEl('consoleLimiterReadout').textContent = `CEILING ${ceilingDb(ceil).toFixed(1)}`;
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
  if (cachedEl('consoleOutputReadout')) cachedEl('consoleOutputReadout').textContent = metrics.lufs_momentary != null ? `${Number(metrics.lufs_momentary).toFixed(1)} LUFS` : (cachedEl('consoleLufs')?.textContent || '-∞ LUFS');
}

export function getChainOverrides() {
  const bypass = studio.getBypassState();
  return { comp_bypass: !!bypass.comp, stereo_bypass: !!bypass.stereo, limiter_bypass: !!bypass.limiter };
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
      syncMetersFromDom(); lastFrame = now;
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
