// js/meters-dashboard.js — Dashboard + live meters + input binding
import { apiFetch, apiBase, getToken } from './api.js';
import * as state from './state.js';

let dashboardPollTimer = null;
let wired = false;

// ── Dashboard ─────────────────────────────────────────────────
function renderDashboard(stats) {
  const $ = (id) => document.getElementById(id);
  if ($('dashCpu')) $('dashCpu').textContent = stats.cpu_percent.toFixed(1) + '%';
  if ($('dashCpuBar')) $('dashCpuBar').style.width = Math.min(100, stats.cpu_percent) + '%';
  if ($('dashRam')) $('dashRam').textContent = stats.rms_percent?.toFixed(1) ?? stats.ram_percent?.toFixed(1) ?? '--' + '%';
  if ($('dashRamBar')) $('dashRamBar').style.width = Math.min(100, stats.ram_percent ?? 0) + '%';
  if ($('dashQueueTotal')) $('dashQueueTotal').textContent = stats.queue?.total ?? 0;
  if ($('dashQueued')) $('dashQueued').textContent = `en cola: ${stats.queue?.queued ?? 0}`;
  if ($('dashProcessing')) $('dashProcessing').textContent = `procesando: ${stats.queue?.processing ?? 0}`;
  if (stats.active_job) {
    const eta = stats.active_job.eta_sec;
    if ($('dashEta')) $('dashEta').textContent = eta != null ? `~${eta}s restante` : 'Procesando…';
    if ($('dashActiveFile')) $('dashActiveFile').textContent = stats.active_job.filename || '';
  } else {
    if ($('dashEta')) $('dashEta').textContent = 'Inactivo';
    if ($('dashActiveFile')) $('dashActiveFile').textContent = '';
  }
}

async function pollDashboard() {
  try {
    const res = await apiFetch(`${apiBase()}/dashboard`);
    if (!res.ok) return;
    renderDashboard(await res.json());
  } catch {}
}

function startDashboard() {
  stopDashboard();
  if (!getToken()) return;
  dashboardPollTimer = setInterval(pollDashboard, 5000);
  pollDashboard();
}

function stopDashboard() {
  if (dashboardPollTimer) { clearInterval(dashboardPollTimer); dashboardPollTimer = null; }
}

// ── Live Meters subscription ──────────────────────────────────
function setupMeters() {
  const clamp01 = (v) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
  const fill = (id, value, floor, ceiling = 0) => {
    const el = document.getElementById(id);
    if (!el) return;
    const n = Number(value);
    el.style.width = `${clamp01((n - floor) / (ceiling - floor)) * 100}%`;
  };
  const text = (id, value, suffix = '') => {
    const el = document.getElementById(id);
    if (!el) return;
    const n = Number(value);
    el.textContent = Number.isFinite(n) ? `${n.toFixed(1)}${suffix}` : `-∞${suffix}`;
  };

  window.addEventListener('lgmdm:metrics', (e) => {
    const m = e.detail?.metrics || {};
    const peak = Number(m.peak_db);
    const rms = Number(m.rms_db);
    const lufs = Number(m.lufs_momentary ?? m.lufs);
    const truePeak = Number(m.true_peak_db);
    const corr = Number(m.stereo_correlation);
    const mono = Number(m.mono_compatibility_db);

    fill('meterPeakFill', peak, -60, 0);
    fill('meterRmsFill', rms, -60, 0);
    fill('meterLufsFill', lufs, -40, 0);
    fill('meterTruePeakFill', truePeak, -60, 0);
    text('meterPeakReadout', peak, ' dB');
    text('meterRmsReadout', rms, ' dB');
    text('meterLufsReadout', lufs, ' LUFS');
    text('meterTruePeakReadout', truePeak, ' dBTP');

    const stereoFill = document.getElementById('stereoMeterFill');
    if (stereoFill && Number.isFinite(corr)) {
      stereoFill.style.width = `${clamp01((corr + 1) / 2) * 100}%`;
    }
    const corrEl = document.getElementById('stereoMeterReadout');
    if (corrEl && Number.isFinite(corr)) corrEl.textContent = `corr: ${corr.toFixed(2)}`;
    const monoEl = document.getElementById('monoCompatReadout');
    if (monoEl && Number.isFinite(mono)) monoEl.textContent = `mono: ${mono.toFixed(1)} dB`;

    const bands = Array.isArray(m.spectrum) ? m.spectrum : [];
    const bandIds = ['sub', 'bass', 'lowmid', 'mid', 'highmid', 'air'];
    bandIds.forEach((id, i) => {
      const v = Number(bands[i]);
      const bar = document.getElementById(`fb-${id}`);
      const read = document.getElementById(`fbv-${id}`);
      if (bar && Number.isFinite(v)) bar.style.width = `${clamp01((v + 80) / 80) * 100}%`;
      if (read && Number.isFinite(v)) read.textContent = `${v.toFixed(0)} dB`;
    });

    // BUGFIX: el motor devuelve los medidores anidados en chain_meters.{mb,comp,
    // glue,parallel} (nombres cortos), no sueltos en la raíz como *_meters —
    // por eso nunca se movían. Se deja el nombre viejo como fallback por si
    // algún caller antiguo todavía manda la forma plana.
    const chain = m.chain_meters || m.chainMeters || {};
    const mb = chain.mb || m.mb_meters || {};
    const compM = chain.comp || m.comp_meters || {};
    const glueM = chain.glue || m.glue_meters || {};
    const parM = chain.parallel || m.parallel_meters || {};
    const hasGrData = [mb.low_gr_db, mb.mid_gr_db, mb.high_gr_db, compM.gr_db].some(Number.isFinite);
    const grSection = document.getElementById('mbGrSection');
    if (grSection && hasGrData) grSection.classList.remove('hidden-panel');

    const grBar = (barId, readId, grDb, opts = {}) => {
      const bar = document.getElementById(barId);
      const read = document.getElementById(readId);
      const v = Number(grDb);
      const reducedDb = Number.isFinite(v) ? Math.abs(v) : 0;
      if (bar) bar.style.width = `${clamp01(reducedDb / 18) * 100}%`;
      if (read) read.textContent = Number.isFinite(v) ? `${reducedDb.toFixed(1)} dB` : (opts.bypassLabel || '0.0 dB');
    };
    grBar('grBarLow', 'grReadLow', mb.low_gr_db);
    grBar('grBarMid', 'grReadMid', mb.mid_gr_db);
    grBar('grBarHigh', 'grReadHigh', mb.high_gr_db);
    grBar('grBarComp', 'grReadComp', compM.gr_db);
    grBar('grBarGlue', 'grReadGlue', glueM.bypass ? null : glueM.gr_db, { bypassLabel: 'bypass' });
    grBar('grBarParallel', 'grReadParallel', parM.bypass ? null : parM.gr_db, { bypassLabel: 'bypass' });
  });
}

// ── Input binding → preview re-render ─────────────────────────
const previewTriggerIds = [
  's-ingain','s-peak','s-uselufs','s-lufstarget','s-thresh','s-ratio',
  's-cattack','s-crelease','s-cmakeup','s-comp-link','s-oversample',
  's-glue-bypass','s-glue-thresh','s-glue-ratio','s-glue-attack','s-glue-release',
  's-glue-makeup','s-glue-pdr','s-glue-pdr-hold','s-hp','s-air','s-shelf-freq',
  's-lowshelf','s-lowshelf-freq','s-comp-pdr','s-comp-pdr-hold',
  's-mb-pdr','s-mb-pdr-hold','s-mscomp-pdr','s-mscomp-pdr-hold',
  's-mb-sw-lowx','s-mb-sw-highx','s-mb-sw-low','s-mb-sw-mid','s-mb-sw-high',
  's-eq1freq','s-eq1gain','s-eq1q','s-eq2freq','s-eq2gain','s-eq2q',
  's-eq3freq','s-eq3gain','s-eq3q','s-eq4freq','s-eq4gain','s-eq4q',
  's-eq5freq','s-eq5gain','s-eq5q','s-eq6freq','s-eq6gain','s-eq6q',
  's-tatt','s-tsus','s-satdrive','s-satmode','s-satmix','s-mgain','s-sgain',
  's-width','s-enhancer','s-haas','s-bassmono','s-rsize','s-rwet',
  's-ceiling','s-lrelease','s-format','s-preview-from',
  's-mb-lowx','s-mb-highx','s-mb-low-th','s-mb-low-ratio','s-mb-low-att','s-mb-low-rel','s-mb-low-mu',
  's-mb-mid-th','s-mb-mid-ratio','s-mb-mid-att','s-mb-mid-rel','s-mb-mid-mu',
  's-mb-high-th','s-mb-high-ratio','s-mb-high-att','s-mb-high-rel','s-mb-high-mu',
  'mb-bypass',
  's-dyneq-bypass','s-dyneq-freq','s-dyneq-q','s-dyneq-thresh','s-dyneq-ratio','s-dyneq-attack','s-dyneq-release','s-dyneq-maxred',
  's-reso-bypass','s-reso-freq','s-reso-q','s-reso-thresh','s-reso-ratio','s-reso-attack','s-reso-release','s-reso-maxred',
  's-mono-freq','s-mono-amount','s-eq-mode','s-lp-taps',
  's-tonalbal-bypass','s-tonalbal-amount','s-tonalbal-boost','s-tonalbal-cut','s-tonalbal-bands',
  'parallelBypass','parallelMix','parallelThresh','parallelRatio','parallelAttack','parallelRelease',
  'mb-stereo-bypass','s-clip-bypass','s-clip-mode','s-clip-ceiling','s-clip-drive',
  's-lp-bypass','s-lp-cutoff','s-mseq-bypass','s-mseq-mid-freq','s-mseq-side-freq',
  's-mscomp-bypass','s-nr-bypass','s-nr-strength','s-nr-noise-sample-sec',
];

function bindInputs() {
  if (wired) return;
  wired = true;
  previewTriggerIds.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const evt = el.tagName === 'SELECT' || el.type === 'checkbox' ? 'change' : 'input';
    el.addEventListener(evt, () => {
      window.dispatchEvent(new CustomEvent('lgmdm:param-change'));
    });
  });
}

export function init() {
  if (init.done) return;
  init.done = true;
  setupMeters();
  bindInputs();
  window.addEventListener('lgmdm:authenticated', startDashboard);
}

export { stopDashboard };
