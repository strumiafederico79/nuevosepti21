// js/studio-controller.js — Rack UI, plugin toggles, bypass, GR meters, chain display
import { cachedEl } from './00-dom-safety.js';

const STORAGE_KEY = 'lgmdm.studio.state.v1';

const PLUGINS = {
  input: {
    family: 'INPUT', label: 'Gain / Trim', short: 'INPUT',
    pinned: true, controls: [
      { key: 'input_gain_db', label: 'Gain', type: 'range', min: -12, max: 12, step: 0.1, input: 's-ingain', suffix: ' dB' }
    ]
  },
  eq: {
    family: 'EQ', label: 'Parametric EQ', short: 'EQ',
    controls: [
      { key: 'eq1_freq', label: 'Band 1 Freq', type: 'range', min: 20, max: 20000, step: 1, input: 's-eq1freq', suffix: ' Hz' },
      { key: 'eq1_gain', label: 'Band 1 Gain', type: 'range', min: -18, max: 18, step: 0.1, input: 's-eq1gain', suffix: ' dB' },
      { key: 'eq1_q', label: 'Band 1 Q', type: 'range', min: 0.1, max: 10, step: 0.01, input: 's-eq1q', suffix: '' },
      { key: 'eq2_freq', label: 'Band 2 Freq', type: 'range', min: 20, max: 20000, step: 1, input: 's-eq2freq', suffix: ' Hz' },
      { key: 'eq2_gain', label: 'Band 2 Gain', type: 'range', min: -18, max: 18, step: 0.1, input: 's-eq2gain', suffix: ' dB' },
      { key: 'eq2_q', label: 'Band 2 Q', type: 'range', min: 0.1, max: 10, step: 0.01, input: 's-eq2q', suffix: '' }
    ]
  },
  dynamic_eq: {
    family: 'EQ', label: 'Dynamic EQ', short: 'DYN EQ', controls: [
      { key: 'dyneq_freq', label: 'Frequency', type: 'range', min: 20, max: 20000, step: 1, input: 's-dyneq-freq', suffix: ' Hz' },
      { key: 'dyneq_q', label: 'Q', type: 'range', min: 0.2, max: 10, step: 0.05, input: 's-dyneq-q', suffix: '' },
      { key: 'dyneq_threshold_db', label: 'Threshold', type: 'range', min: -60, max: 0, step: 0.5, input: 's-dyneq-thresh', suffix: ' dB' },
      { key: 'dyneq_ratio', label: 'Ratio', type: 'range', min: 1, max: 20, step: 0.1, input: 's-dyneq-ratio', suffix: ':1' },
      { key: 'dyneq_attack_ms', label: 'Attack', type: 'range', min: 0.1, max: 200, step: 0.1, input: 's-dyneq-attack', suffix: ' ms' },
      { key: 'dyneq_release_ms', label: 'Release', type: 'range', min: 10, max: 1000, step: 5, input: 's-dyneq-release', suffix: ' ms' }
    ]
  },
  compressor: {
    family: 'DYNAMICS', label: 'Compressor', short: 'COMP', controls: [
      { key: 'comp_threshold_db', label: 'Threshold', type: 'range', min: -60, max: 0, step: 0.5, input: 's-thresh', suffix: ' dB' },
      { key: 'comp_ratio', label: 'Ratio', type: 'range', min: 1, max: 20, step: 0.1, input: 's-ratio', suffix: ':1' },
      { key: 'comp_attack_ms', label: 'Attack', type: 'range', min: 0.1, max: 200, step: 0.1, input: 's-cattack', suffix: ' ms' },
      { key: 'comp_release_ms', label: 'Release', type: 'range', min: 10, max: 1000, step: 5, input: 's-crelease', suffix: ' ms' },
      { key: 'comp_makeup_db', label: 'Make-up', type: 'range', min: -12, max: 24, step: 0.5, input: 's-cmakeup', suffix: ' dB' }
    ], stageBypass: 'comp'
  },
  multiband: {
    family: 'DYNAMICS', label: 'Comp-Multiband', short: 'MB COMP', controls: [
      { key: 'mb_low_x', label: 'Low → Mid', type: 'range', min: 20, max: 2000, step: 10, input: 's-mb-lowx', suffix: ' Hz', global: true },
      { key: 'mb_high_x', label: 'Mid → High', type: 'range', min: 500, max: 12000, step: 10, input: 's-mb-highx', suffix: ' Hz', global: true }
    ], bands: {
      low: [
        { key: 'mb_low_threshold_db', label: 'Threshold', input: 's-mb-low-th', suffix: ' dB', min: -60, max: 0, step: 0.5 },
        { key: 'mb_low_ratio', label: 'Ratio', input: 's-mb-low-ratio', suffix: ':1', min: 1, max: 20, step: 0.1 },
        { key: 'mb_low_attack_ms', label: 'Attack', input: 's-mb-low-att', suffix: ' ms', min: 0.1, max: 200, step: 0.1 },
        { key: 'mb_low_release_ms', label: 'Release', input: 's-mb-low-rel', suffix: ' ms', min: 10, max: 1000, step: 5 },
        { key: 'mb_low_makeup_db', label: 'Make-up', input: 's-mb-low-mu', suffix: ' dB', min: -12, max: 24, step: 0.5 }
      ],
      mid: [
        { key: 'mb_mid_threshold_db', label: 'Threshold', input: 's-mb-mid-th', suffix: ' dB', min: -60, max: 0, step: 0.5 },
        { key: 'mb_mid_ratio', label: 'Ratio', input: 's-mb-mid-ratio', suffix: ':1', min: 1, max: 20, step: 0.1 },
        { key: 'mb_mid_attack_ms', label: 'Attack', input: 's-mb-mid-att', suffix: ' ms', min: 0.1, max: 200, step: 0.1 },
        { key: 'mb_mid_release_ms', label: 'Release', input: 's-mb-mid-rel', suffix: ' ms', min: 10, max: 1000, step: 5 },
        { key: 'mb_mid_makeup_db', label: 'Make-up', input: 's-mb-mid-mu', suffix: ' dB', min: -12, max: 24, step: 0.5 }
      ],
      high: [
        { key: 'mb_high_threshold_db', label: 'Threshold', input: 's-mb-high-th', suffix: ' dB', min: -60, max: 0, step: 0.5 },
        { key: 'mb_high_ratio', label: 'Ratio', input: 's-mb-high-ratio', suffix: ':1', min: 1, max: 20, step: 0.1 },
        { key: 'mb_high_attack_ms', label: 'Attack', input: 's-mb-high-att', suffix: ' ms', min: 0.1, max: 200, step: 0.1 },
        { key: 'mb_high_release_ms', label: 'Release', input: 's-mb-high-rel', suffix: ' ms', min: 10, max: 1000, step: 5 },
        { key: 'mb_high_makeup_db', label: 'Make-up', input: 's-mb-high-mu', suffix: ' dB', min: -12, max: 24, step: 0.5 }
      ]
    }, bypassInput: 'mb-bypass'
  },
  transient: {
    family: 'DYNAMICS', label: 'Transient Shaper', short: 'TRANSIENT',
    controls: [
      { key: 'transient_attack', label: 'Attack', input: 's-tatt', suffix: ' %', min: -100, max: 100, step: 1 },
      { key: 'transient_sustain', label: 'Sustain', input: 's-tsus', suffix: ' %', min: -100, max: 100, step: 1 }
    ]
  },
  glue: {
    family: 'DYNAMICS', label: 'Glue Compressor', short: 'GLUE', controls: [
      { key: 'glue_threshold_db', label: 'Threshold', input: 's-glue-thresh', suffix: ' dB', min: -24, max: 0, step: 0.5 },
      { key: 'glue_ratio', label: 'Ratio', input: 's-glue-ratio', suffix: ':1', min: 1, max: 10, step: 0.5 },
      { key: 'glue_attack_ms', label: 'Attack', input: 's-glue-attack', suffix: ' ms', min: 0.1, max: 200, step: 0.1 },
      { key: 'glue_release_ms', label: 'Release', input: 's-glue-release', suffix: ' ms', min: 10, max: 1000, step: 5 },
      { key: 'glue_makeup_db', label: 'Make-up', input: 's-glue-makeup', suffix: ' dB', min: -12, max: 12, step: 0.5 }
    ], bypassInput: 's-glue-bypass'
  },
  ms_comp: {
    family: 'STEREO', label: 'M/S Compressor', short: 'M/S COMP', controls: [
      { key: 'ms_comp_mid_threshold_db', label: 'Mid Threshold', input: 's-mscomp-mid-thresh', suffix: ' dB', min: -60, max: 0, step: 0.5 },
      { key: 'ms_comp_mid_ratio', label: 'Mid Ratio', input: 's-mscomp-mid-ratio', suffix: ':1', min: 1, max: 20, step: 0.5 },
      { key: 'ms_comp_side_threshold_db', label: 'Side Threshold', input: 's-mscomp-side-thresh', suffix: ' dB', min: -60, max: 0, step: 0.5 },
      { key: 'ms_comp_side_ratio', label: 'Side Ratio', input: 's-mscomp-side-ratio', suffix: ':1', min: 1, max: 20, step: 0.5 },
      { key: 'ms_comp_side_attack_ms', label: 'Side Attack', input: 's-mscomp-side-attack', suffix: ' ms', min: 0.1, max: 200, step: 0.1 },
      { key: 'ms_comp_side_release_ms', label: 'Side Release', input: 's-mscomp-side-release', suffix: ' ms', min: 5, max: 2000, step: 5 }
    ], bypassInput: 's-mscomp-bypass'
  },
  saturation: {
    family: 'COLOR', label: 'Saturation', short: 'SAT', controls: [
      { key: 'saturation_drive', label: 'Drive', input: 's-satdrive', suffix: ' dB', min: 0, max: 24, step: 0.1 },
      { key: 'saturation_mix', label: 'Mix', input: 's-satmix', suffix: ' %', min: 0, max: 1, step: 0.01, displayScale: 100 }
    ]
  },
  stereo: {
    family: 'STEREO', label: 'Stereo Width', short: 'STEREO', controls: [
      { key: 'stereo_width_amount', label: 'Width', input: 's-width', suffix: '×', min: 0.5, max: 1.5, step: 0.01 }
    ], stageBypass: 'stereo'
  },
  mb_stereo: {
    family: 'STEREO', label: 'Multiband Stereo', short: 'MB STEREO', controls: [
      { key: 'mb_stereo_low_width', label: 'Low width', input: 's-mb-sw-low', suffix: '×', min: 0, max: 2, step: 0.01 },
      { key: 'mb_stereo_mid_width', label: 'Mid width', input: 's-mb-sw-mid', suffix: '×', min: 0, max: 2, step: 0.01 },
      { key: 'mb_stereo_high_width', label: 'High width', input: 's-mb-sw-high', suffix: '×', min: 0, max: 2, step: 0.01 }
    ], bypassInput: 'mb-stereo-bypass'
  },
  clipper: {
    family: 'OUTPUT', label: 'Clipper', short: 'CLIP', controls: [
      { key: 'clipper_ceiling', label: 'Ceiling', input: 's-clip-ceiling', suffix: ' dB', min: -6, max: 0, step: 0.1 },
      { key: 'clipper_drive_db', label: 'Drive', input: 's-clip-drive', suffix: ' dB', min: 0, max: 12, step: 0.1 }
    ], bypassInput: 's-clip-bypass'
  },
  limiter: {
    family: 'OUTPUT', label: 'True Peak Limiter', short: 'LIMITER', controls: [
      { key: 'limiter_ceiling', label: 'Ceiling', input: 's-ceiling', suffix: ' dB', min: -3, max: -0.1, step: 0.1 },
      { key: 'limiter_release_ms', label: 'Release', input: 's-lrelease', suffix: ' ms', min: 10, max: 1000, step: 5 }
    ], stageBypass: 'limiter'
  }
};

const FAMILY_ORDER = ['INPUT', 'EQ', 'DYNAMICS', 'COLOR', 'STEREO', 'OUTPUT'];
const BAND_ORDER = ['low', 'mid', 'high'];

const state = {
  active: new Set(['input']),
  bypassed: new Set(),
  expanded: 'input',
  multibandBand: 'mid',
  savedValues: {},
  mounted: false
};

function getInput(inputId) {
  const el = cachedEl(inputId);
  if (!el) throw new Error(`[Studio] Falta control técnico: #${inputId}`);
  return el;
}

export function read(inputId) {
  return getInput(inputId).value;
}

export function set(inputId, value) {
  const el = getInput(inputId);
  el.value = String(value);
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

export function formatValue(value, def) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  const scaled = def.displayScale ? n * def.displayScale : n;
  const digits = Math.abs(scaled - Math.round(scaled)) < 1e-7 ? 0 : 2;
  return `${scaled.toFixed(digits)}${def.suffix || ''}`;
}

function saveStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      active: [...state.active], expanded: state.expanded,
      band: state.multibandBand, savedValues: state.savedValues
    }));
  } catch (_) {}
}

function loadStorage() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (!data) return;
    if (Array.isArray(data.active)) data.active.forEach(k => { if (PLUGINS[k]) state.active.add(k); });
    if (Array.isArray(data.removed)) data.removed.forEach(k => state.active.delete(k));
    if (PLUGINS[data.expanded]) state.expanded = data.expanded;
    if (BAND_ORDER.includes(data.band)) state.multibandBand = data.band;
    if (data.savedValues && typeof data.savedValues === 'object') state.savedValues = data.savedValues;
  } catch (_) {}
}

export function isActive(key) {
  return state.active.has(key);
}

export function isStageBypassed(stage) {
  const plugin = Object.entries(PLUGINS).find(([, p]) => p.stageBypass === stage)?.[0];
  if (!plugin) return false;
  return state.bypassed.has(plugin) || !state.active.has(plugin);
}

export function getBypassState() {
  const result = {};
  for (const [key, plugin] of Object.entries(PLUGINS)) {
    if (plugin.stageBypass) {
      result[plugin.stageBypass] = state.bypassed.has(key) || !state.active.has(key);
    }
  }
  return result;
}

export function setPluginBypass(key, bypassed) {
  const plugin = PLUGINS[key];
  if (!plugin) return;
  if (bypassed) state.bypassed.add(key); else state.bypassed.delete(key);
  saveStorage();
  window.dispatchEvent(new CustomEvent('lgmdm:param-change'));
}

export function syncPluginBypass(key, enabled) {
  setPluginBypass(key, !enabled);
}

export function activate(key, enabled) {
  const plugin = PLUGINS[key];
  if (!plugin) return;
  if (plugin.pinned && !enabled) return;
  if (enabled) {
    state.active.add(key);
    state.bypassed.delete(key);
  } else {
    state.active.delete(key);
    state.bypassed.add(key);
  }
  state.expanded = enabled ? key : (state.expanded === key ? null : state.expanded);
  render();
  saveStorage();
  window.dispatchEvent(new CustomEvent('lgmdm:studio-chain-changed', {
    detail: { active: [...state.active], plugin: key, enabled }
  }));
  window.dispatchEvent(new CustomEvent('lgmdm:param-change'));
}

const FAMILY_COLORS = { INPUT: 'var(--text-muted)', EQ: 'var(--accent2, #06b6d4)', DYNAMICS: 'var(--yellow, #eab308)', COLOR: 'var(--red, #ef4444)', STEREO: '#a78bfa', OUTPUT: 'var(--green, #22c55e)' };

function renderDspList() {
  const mount = cachedEl('dsp-plugins-mount');
  if (!mount) return;
  mount.innerHTML = '';
  FAMILY_ORDER.forEach(family => {
    const block = document.createElement('div');
    block.className = 'dsp-family-group';
    const title = document.createElement('div');
    title.className = 'dsp-family-title';
    title.innerHTML = `<span style="color:${FAMILY_COLORS[family] || 'var(--text-muted)'}">${family}</span>`;
    block.appendChild(title);
    Object.entries(PLUGINS).filter(([, p]) => p.family === family).forEach(([key, p]) => {
      const row = document.createElement('label');
      row.className = `dsp-plugin-row ${state.active.has(key) ? 'active' : ''}`;
      row.innerHTML = `<input type="checkbox" ${state.active.has(key) ? 'checked' : ''} ${p.pinned ? 'disabled' : ''} data-dsp-toggle="${key}"><span class="dsp-plugin-led" style="background:${FAMILY_COLORS[family] || 'var(--text-muted)'}"></span><span class="dsp-plugin-name">${p.label}</span>${p.pinned ? '<em class="dsp-badge">CORE</em>' : ''}`;
      row.querySelector('input').addEventListener('change', e => activate(key, e.target.checked));
      block.appendChild(row);
    });
    mount.appendChild(block);
  });
}

function cardHeader(key) {
  const p = PLUGINS[key];
  const open = state.expanded === key;
  return `<button class="studio-plugin-header" type="button" data-studio-expand="${key}" aria-expanded="${open}"><span class="studio-plugin-status"></span><span class="studio-plugin-header-title">${p.label}</span><span class="studio-plugin-header-meta">${p.short}</span><span class="studio-chevron">${open ? '▼' : '▶'}</span></button>`;
}

function renderControls(key) {
  const p = PLUGINS[key];
  const defs = [];
  if (p.controls) defs.push(...p.controls);
  if (key === 'multiband') defs.push(...p.bands[state.multibandBand]);
  const wrap = document.createElement('div');
  wrap.className = 'studio-plugin-body';
  if (key === 'multiband') {
    const bandNav = document.createElement('div');
    bandNav.className = 'studio-band-tabs';
    BAND_ORDER.forEach(band => {
      const btn = document.createElement('button');
      btn.type = 'button'; btn.className = `studio-band-tab ${band === state.multibandBand ? 'active' : ''}`; btn.textContent = band.toUpperCase();
      btn.addEventListener('click', () => { state.multibandBand = band; saveStorage(); render(); });
      bandNav.appendChild(btn);
    });
    wrap.appendChild(bandNav);
  }
  defs.forEach(def => {
    const source = getInput(def.input);
    const row = document.createElement('label');
    row.className = 'studio-control-row';
    row.innerHTML = `<span class="studio-control-label">${def.label}</span><input type="range" min="${def.min}" max="${def.max}" step="${def.step}" value="${source.value}" class="studio-range" data-studio-input="${def.input}"><output>${formatValue(source.value, def)}</output>`;
    const range = row.querySelector('input'); const output = row.querySelector('output');
    range.addEventListener('input', () => { set(def.input, range.value); output.textContent = formatValue(range.value, def); });
    wrap.appendChild(row);
  });
  const actions = document.createElement('div'); actions.className = 'studio-plugin-actions';
  const bypass = document.createElement('button'); bypass.type = 'button'; bypass.className = 'studio-action-btn'; bypass.textContent = 'BYPASS';
  bypass.addEventListener('click', () => {
    const currentlyBypassed = state.bypassed.has(key);
    setPluginBypass(key, !currentlyBypassed);
    render(); saveStorage();
  });
  const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'studio-action-btn danger'; remove.textContent = 'REMOVE';
  remove.addEventListener('click', () => activate(key, false));
  actions.appendChild(bypass); actions.appendChild(remove); wrap.appendChild(actions);
  return wrap;
}

function renderRack() {
  const mount = cachedEl('studioRackMount');
  if (!mount) return;
  mount.innerHTML = '';
  [...state.active].forEach(key => {
    const card = document.createElement('article'); card.className = `studio-plugin-card ${state.expanded === key ? 'expanded' : ''}`;
    card.innerHTML = cardHeader(key);
    card.querySelector('[data-studio-expand]').addEventListener('click', () => { state.expanded = state.expanded === key ? null : key; saveStorage(); renderRack(); });
    if (state.expanded === key) card.appendChild(renderControls(key));
    mount.appendChild(card);
  });
  const add = document.createElement('button'); add.type = 'button'; add.className = 'studio-add-plugin'; add.textContent = '+ ADD PLUGIN';
  add.addEventListener('click', () => {
    const candidate = Object.keys(PLUGINS).find(k => !state.active.has(k));
    if (candidate) activate(candidate, true);
  });
  mount.appendChild(add);
}

function renderCenter() {
  const chain = cachedEl('studioCenterChain');
  if (!chain) return;
  chain.innerHTML = '';
  [...state.active].forEach((key, i) => {
    const p = PLUGINS[key];
    const node = document.createElement('div'); node.className = 'studio-chain-node';
    node.innerHTML = `<span class="studio-chain-index">${String(i + 1).padStart(2, '0')}</span><span class="studio-chain-name">${p.short}</span>${i < state.active.size - 1 ? '<span class="studio-chain-arrow">›</span>' : ''}`;
    chain.appendChild(node);
  });
}

export function render() {
  renderDspList();
  renderRack();
  renderCenter();
  document.body.classList.add('lg-studio-active');
}

export function setGRValues(values) {
  ['low', 'mid', 'high'].forEach(band => {
    const value = Number(values?.[band] ?? 0);
    const out = cachedEl(`studioGr_${band}`); const bar = cachedEl(`studioGrBar_${band}`);
    if (out) out.textContent = Number.isFinite(value) ? `${value.toFixed(1)} dB` : '-- dB';
    if (bar) bar.style.width = Number.isFinite(value) ? `${Math.min(100, Math.max(0, Math.abs(value) * 8))}%` : '0%';
  });
}

let previewTelemetry = null;
let telemetryAudio = null;
let telemetryTick = null;
let telemetryEnded = null;

export function telemetryAt(timeSec) {
  const curves = previewTelemetry?.multiband?.bands;
  const duration = Number(previewTelemetry?.duration_sec);
  if (!Array.isArray(curves) || curves.length !== 3 || !Number.isFinite(duration) || duration <= 0) return null;
  const values = {};
  for (const band of ['low', 'mid', 'high']) {
    const arr = curves.find(x => x.band === band)?.gain_reduction_db;
    if (!arr?.length) return null;
    const pos = Math.min(arr.length - 1, Math.max(0, Math.round((timeSec / duration) * (arr.length - 1))));
    values[band] = Number(arr[pos]);
  }
  return values;
}

export function updateCentralGR(metrics) {
  const panel = cachedEl('studioGrPanel');
  if (!panel) return;
  panel.hidden = false;
  const mb = metrics?.mb_meters || metrics?.chain_meters?.mb_meters || {};
  const comp = metrics?.comp_meters || metrics?.chain_meters?.comp || {};
  const glue = metrics?.glue_meters || metrics?.chain_meters?.glue || {};
  const values = { comp: Number(comp.gr_db), low: Number(mb.low_gr_db), mid: Number(mb.mid_gr_db), high: Number(mb.high_gr_db), glue: Number(glue.gr_db) };
  [['comp', 'studioGrBar_comp', 'studioGr_comp'], ['low', 'studioGrBar_low', 'studioGr_low'], ['mid', 'studioGrBar_mid', 'studioGr_mid'], ['high', 'studioGrBar_high', 'studioGr_high'], ['glue', 'studioGrBar_glue', 'studioGr_glue']].forEach(([key, barId, outId]) => {
    const value = values[key]; const finite = Number.isFinite(value);
    const out = cachedEl(outId); const bar = cachedEl(barId);
    if (out) out.textContent = finite ? `${value.toFixed(1)} dB` : '-- dB';
    if (bar) bar.style.width = finite ? `${Math.min(100, Math.max(0, Math.abs(value) * 8))}%` : '0%';
  });
}

export function install() {
  if (state.mounted) return;
  localStorage.removeItem(STORAGE_KEY);
  state.active = new Set(['input']);
  state.bypassed = new Set(Object.keys(PLUGINS).filter(key => key !== 'input'));
  state.expanded = 'input';
  state.savedValues = {};
  saveStorage();
  render();
  window.addEventListener('lgmdm:metrics', e => updateCentralGR(e.detail?.metrics));
  window.addEventListener('lgmdm:preview-telemetry', e => {
    previewTelemetry = e.detail?.telemetry || null;
    const audio = e.detail?.audio || document.querySelector('#previewAudioWrap audio');
    if (!audio || telemetryAudio === audio) return;
    if (telemetryAudio && telemetryTick) {
      telemetryAudio.removeEventListener('timeupdate', telemetryTick);
      if (telemetryEnded) telemetryAudio.removeEventListener('ended', telemetryEnded);
    }
    telemetryAudio = audio;
    telemetryTick = () => { if (!telemetryAudio?.paused) { const v = telemetryAt(telemetryAudio.currentTime); if (v) setGRValues(v); } };
    telemetryEnded = () => setGRValues(null);
    audio.addEventListener('timeupdate', telemetryTick);
    audio.addEventListener('ended', telemetryEnded);
  });
  state.mounted = true;
}

export function getActiveChain() { return [...state.active]; }
export function getBypassedPlugins() { return Object.keys(PLUGINS).filter(key => !state.active.has(key) || state.bypassed.has(key)); }
export function getPlugins() { return PLUGINS; }
