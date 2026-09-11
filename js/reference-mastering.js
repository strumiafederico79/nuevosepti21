// js/reference-mastering.js — Reference matching, band EQ, preview, polling, A/B
import { cachedEl } from './00-dom-safety.js';
import { apiFetch, apiBase, downloadAuthenticated } from './api.js';
import { showToast, showProgress, normalizeError } from './00-error-handler.js';

const referenceState = { file: null, libraryId: null };
let currentJobId = null;
let downloadUrl = '';

// ── collectReferenceParamsObj ─────────────────────────────────
export function collectReferenceParamsObj() {
  return {
    eq_max_boost_db: cachedEl('s-ref-eq')?.checked ? cachedEl('s-ref-boost')?.value : '0',
    eq_max_cut_db: cachedEl('s-ref-eq')?.checked ? cachedEl('s-ref-cut')?.value : '0',
    eq_fit_method: cachedEl('s-ref-eqmethod')?.value || 'heuristic',
    match_loudness: cachedEl('s-ref-loudness')?.checked ?? true,
    match_dynamics: cachedEl('s-ref-dynamics')?.checked ?? true,
    match_stereo_width: cachedEl('s-ref-stereo')?.checked ?? true,
    match_transient: cachedEl('s-ref-transient')?.checked ?? true,
    match_sub_bass: cachedEl('s-ref-subbass')?.checked ?? false,
    match_desser: cachedEl('s-ref-desser')?.checked ?? false,
    match_saturation: cachedEl('s-ref-saturation')?.checked ?? false,
    output_format: cachedEl('s-format')?.value || 'wav',
    output_bit_depth: cachedEl('s-bitdepth')?.value || '24',
    dynamics_margin_db: cachedEl('s-ref-dynmargin')?.value || '1',
    stereo_blend: ((parseFloat(cachedEl('s-ref-stereoblend')?.value || '100') / 100)).toFixed(2),
    ms_eq_matching: cachedEl('s-ref-ms-eq')?.checked ?? true,
    iterative_eq_passes: parseInt(cachedEl('s-ref-eq-passes')?.value || '3'),
    match_crest: cachedEl('s-ref-match-crest')?.checked ?? true,
    crest_amount: (parseFloat(cachedEl('s-ref-crest-amount')?.value || '75') / 100).toFixed(2),
    match_spectral_dynamics: cachedEl('s-ref-spectral-dynamics')?.checked ?? true,
    spectral_dynamics_amount: (parseFloat(cachedEl('s-ref-spectral-dyn-amount')?.value || '60') / 100).toFixed(2),
    spectral_dynamics_bins: parseInt(cachedEl('s-ref-spectral-dyn-bins')?.value || '4'),
    use_parallel_compression: cachedEl('s-ref-parallel-comp')?.checked ?? true,
    parallel_mix: (parseFloat(cachedEl('s-ref-parallel-mix')?.value || '28') / 100).toFixed(2),
    parallel_threshold_db: parseFloat(cachedEl('s-ref-parallel-thr')?.value || '-20'),
    parallel_ratio: parseFloat(cachedEl('s-ref-parallel-ratio')?.value || '4'),
    parallel_makeup_db: parseFloat(cachedEl('s-ref-parallel-makeup')?.value || '6'),
    use_multiband_saturation: cachedEl('s-ref-mb-sat')?.checked ?? true,
    mb_sat_mix: (parseFloat(cachedEl('s-ref-mb-sat-mix')?.value || '45') / 100).toFixed(2),
    use_two_stage_limiter: cachedEl('s-ref-two-stage-lim')?.checked ?? true,
    gentle_ceiling_db: parseFloat(cachedEl('s-ref-gentle-ceil')?.value || '-2.5'),
    gentle_release_ms: parseFloat(cachedEl('s-ref-gentle-rel')?.value || '120'),
    max_target_lufs: parseFloat(cachedEl('s-ref-max-lufs')?.value || '-12'),
  };
}

// ── submitReferenceMasterJob ──────────────────────────────────
export async function submitReferenceMasterJob(selectedFile) {
  showProgress('processing', 'Enviando archivos…');
  const btn = cachedEl('btn-master');
  if (btn) btn.disabled = true;
  const fd = new FormData();
  fd.append('file', selectedFile);
  if (referenceState.libraryId) fd.append('reference_library_id', referenceState.libraryId);
  else if (referenceState.file) fd.append('reference_file', referenceState.file);
  const params = new URLSearchParams(collectReferenceParamsObj());
  try {
    const url = `${apiBase()}/master/reference?${params.toString()}`;
    const res = await apiFetch(url, { method: 'POST', body: fd });
    if (!res.ok) { const text = await res.text(); throw new Error(`HTTP ${res.status}: ${text}`); }
    const data = await res.json();
    currentJobId = data.job_id;
    showProgress('queued', `Job ${currentJobId.slice(0, 8)}… en cola`);
    startReferencePolling(currentJobId);
  } catch (e) {
    showProgress('error', 'Error: ' + normalizeError(e));
    if (btn) btn.disabled = false;
  }
}

// ── refBandEQ ────────────────────────────────────────────────
const bandEQState = { bands: [], count: 7 };
const MIN_HZ = 20, MAX_HZ = 20000;

function logFreqs(n) {
  return Array.from({ length: n }, (_, i) => MIN_HZ * Math.pow(MAX_HZ / MIN_HZ, i / (n - 1)));
}

function interpolateBands(oldBands, newFreqs) {
  if (!oldBands.length) return newFreqs.map(() => 0);
  return newFreqs.map(f => {
    const logF = Math.log10(f);
    const logFs = oldBands.map(b => Math.log10(b.freq_hz));
    if (logF <= logFs[0]) return oldBands[0].gain_db;
    if (logF >= logFs[logFs.length - 1]) return oldBands[logFs.length - 1].gain_db;
    for (let i = 0; i < logFs.length - 1; i++) {
      if (logF >= logFs[i] && logF <= logFs[i + 1]) {
        const t = (logF - logFs[i]) / (logFs[i + 1] - logFs[i]);
        return oldBands[i].gain_db * (1 - t) + oldBands[i + 1].gain_db * t;
      }
    }
    return 0;
  });
}

function fmtHz(hz) {
  if (hz >= 1000) return (hz / 1000).toFixed(hz >= 10000 ? 0 : 1) + ' kHz';
  return Math.round(hz) + ' Hz';
}

function renderBands(n, interpolatedGains) {
  const container = cachedEl('ref-band-controls');
  if (!container) return;
  const freqs = logFreqs(n);
  container.innerHTML = '';
  bandEQState.bands = [];
  freqs.forEach((freq, i) => {
    const gain = interpolatedGains ? Math.round(interpolatedGains[i] * 2) / 2 : 0;
    bandEQState.bands.push({ freq_hz: freq, gain_db: gain });
    const div = document.createElement('div');
    div.className = 'ref-band-item';
    const gainStr = (gain >= 0 ? '+' : '') + gain.toFixed(1) + ' dB';
    div.innerHTML = `<label>${fmtHz(freq)}</label><span class="val">${gainStr}</span><input type="range" min="-12" max="12" step="0.5" value="${gain}">`;
    container.appendChild(div);
    const sl = div.querySelector('input'); const val = div.querySelector('span.val');
    sl.addEventListener('input', () => {
      const v = parseFloat(sl.value);
      bandEQState.bands[i].gain_db = v;
      val.textContent = (v >= 0 ? '+' : '') + v.toFixed(1) + ' dB';
      container.dispatchEvent(new CustomEvent('bandchange', { bubbles: true }));
    });
  });
}

export function setBandCount(n, skipInterp = false) {
  const newFreqs = logFreqs(n);
  const gains = skipInterp ? null : interpolateBands(bandEQState.bands, newFreqs);
  renderBands(n, gains);
  bandEQState.count = n;
  const valEl = cachedEl('v-band-count');
  if (valEl) valEl.textContent = n;
}

export function getGainsArray() {
  return bandEQState.bands.map(b => ({ freq_hz: Math.round(b.freq_hz * 10) / 10, gain_db: b.gain_db }));
}

// ── drawEqCurve ──────────────────────────────────────────────
export function drawEqCurve(curve, canvasId = 'refEqCurveCanvas', wrapId = 'refEqCurveWrap') {
  const wrap = cachedEl(wrapId); const canvas = cachedEl(canvasId);
  if (!wrap || !canvas || !curve?.length) return;
  wrap.hidden = false; wrap.style.display = 'block';
  const ctx = canvas.getContext('2d'); const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const ZERO_Y = H / 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.beginPath(); ctx.moveTo(0, ZERO_Y); ctx.lineTo(W, ZERO_Y); ctx.stroke();
  const gains = curve.map(p => p.gain_db); const MAX_G = Math.max(6, ...gains.map(Math.abs));
  ctx.beginPath(); ctx.strokeStyle = '#06b6d4'; ctx.lineWidth = 1.5;
  curve.forEach((p, i) => {
    const x = (i / (curve.length - 1)) * W; const y = ZERO_Y - (p.gain_db / MAX_G) * (H * 0.42);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ── updateMetrics ────────────────────────────────────────────
export function updateMetrics(m) {
  const set = (id, v) => { const el = cachedEl(id); if (el) el.textContent = v; };
  set('rp-lufs', m.lufs_momentary != null ? m.lufs_momentary.toFixed(1) + ' LUFS' : '--');
  set('rp-peak', m.peak_db != null ? m.peak_db.toFixed(1) + ' dBFS' : '--');
  set('rp-rms', m.rms_db != null ? m.rms_db.toFixed(1) + ' dB' : '--');
  set('rp-corr', m.stereo_correlation != null ? m.stereo_correlation.toFixed(2) : '--');
}

// ── startReferencePolling ────────────────────────────────────
export function startReferencePolling(jobId) {
  const poll = setInterval(async () => {
    try {
      const res = await apiFetch(`${apiBase()}/job/${jobId}`);
      const data = await res.json();
      if (data.status === 'queued') showProgress('queued', 'En cola…');
      else if (data.status === 'processing') showProgress('processing', 'Masterizando…', data.progress);
      else if (data.status === 'done') {
        clearInterval(poll);
        showProgress('done', 'Masterizado por referencia ✓');
        downloadUrl = `${apiBase()}/download/${jobId}`;
        const btn = cachedEl('btn-download');
        if (btn) { btn.style.display = 'block'; btn.onclick = () => downloadAuthenticated(downloadUrl, { filename: 'reference-master.wav' }); }
        if (data.reference_match) renderReferenceMatch(data.reference_match, data.analysis_reference, data.analysis_after);
      } else if (data.status === 'error') {
        clearInterval(poll); showProgress('error', 'Error: ' + data.error);
      }
    } catch (e) { console.error('Poll error:', e); }
  }, 1500);
}

// ── _matchBarRow ─────────────────────────────────────────────
function matchBarRow(label, ownVal, refVal, unit, closeThresholdAbs, fmt) {
  fmt = fmt || (v => v);
  if (ownVal == null || refVal == null) return '';
  const diff = Math.abs(ownVal - refVal); const ok = diff <= closeThresholdAbs;
  const lo = Math.min(ownVal, refVal, 0) - Math.abs(refVal || 1) * 0.15;
  const hi = Math.max(ownVal, refVal, 0) + Math.abs(refVal || 1) * 0.15;
  const range = hi - lo || 1;
  const ownPct = Math.max(0, Math.min(100, ((ownVal - lo) / range) * 100));
  const refPct = Math.max(0, Math.min(100, ((refVal - lo) / range) * 100));
  return `<div class="match-bar-row"><div class="match-bar-label">${ok ? '✓' : '⚠'} ${label}</div><div class="match-bar-track"><div class="match-bar-marker match-bar-ref" style="left:${refPct}%" title="Referencia: ${fmt(refVal)}${unit}"></div><div class="match-bar-fill" style="width:${ownPct}%"></div></div><div class="match-bar-values">${fmt(ownVal)}${unit} <span>vs ref ${fmt(refVal)}${unit}</span></div></div>`;
}

// ── renderReferenceMatch ─────────────────────────────────────
export function renderReferenceMatch(rm, refAnalysis, ownAnalysis) {
  const panel = document.createElement('div');
  panel.className = 'ref-match-panel';
  const pct = rm.after?.match_percent ?? 0;
  const report = rm.intelligent_report || {};
  const loudnessBar = matchBarRow('Loudness (LUFS)', ownAnalysis?.lufs, refAnalysis?.lufs, ' LUFS', 0.5, v => v.toFixed(1));
  const dynRows = ['low', 'mid', 'high'].map(name => {
    const b = rm.dynamics_by_band?.[name]; if (!b) return '';
    const label = name === 'low' ? 'Graves' : name === 'mid' ? 'Medios' : 'Agudos';
    return matchBarRow(`${label} (crest)`, b.own_crest_db, b.ref_crest_db, ' dB', 1.5, v => v?.toFixed(1) ?? '--');
  }).join('');
  const tipsHtml = (report.tips || []).map(t => `<li>${t}</li>`).join('');
  const issuesHtml = (report.issues || []).map(t => `<li class="issue">${t}</li>`).join('');
  panel.innerHTML = `
    <h3>🎯 Match con referencia</h3>
    <div class="ref-match-score-row">
      <div class="ref-match-score-circle"><span class="score-num">${pct}%</span><span class="score-label">MATCH</span></div>
      <div>${report.overall_score !== undefined ? `<div>Puntaje: <b>${report.overall_score}/100 (${report.grade})</b></div>` : ''}</div>
    </div>
    <div class="section-title">Loudness</div>
    ${loudnessBar}
    <div class="ref-match-step">Ganancia: <b>${rm.loudness_gain_applied_db >= 0 ? '+' : ''}${rm.loudness_gain_applied_db} dB</b></div>
    <div class="section-title">Dinámica por banda</div>
    ${dynRows}
    ${issuesHtml ? `<ul class="issues">${issuesHtml}</ul>` : ''}
    ${tipsHtml ? `<ul class="tips">${tipsHtml}</ul>` : ''}`;
  const content = cachedEl('results-area') || document.querySelector('#content');
  if (content) content.appendChild(panel);
}

// ── A/B Panel ────────────────────────────────────────────────
let _abSnapshotA = null, _abSnapshotB = null, _abCurrentUrl = null;

export function captureAB(slot, selectedFile, buildParamsFn) {
  if (!selectedFile) { showToast('Seleccioná un archivo primero', 'warning'); return; }
  const status = cachedEl('abStatus');
  if (status) status.textContent = `Capturando ${slot}…`;
  const fd = new FormData(); fd.append('file', selectedFile);
  const params = buildParamsFn ? buildParamsFn() : new URLSearchParams();
  params.set('preview_seconds', '10');
  apiFetch(`${apiBase()}/preview?${params.toString()}`, { method: 'POST', body: fd })
    .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.blob(); })
    .then(blob => {
      if (slot === 'A') { _abSnapshotA = { blob, label: 'A' }; cachedEl('abPlayA').disabled = false; }
      else { _abSnapshotB = { blob, label: 'B' }; cachedEl('abPlayB').disabled = false; }
      if (status) status.textContent = `${slot} capturado ✓`;
    })
    .catch(e => { if (status) status.textContent = 'Error: ' + e.message; });
}

export function playAB(slot) {
  const snap = slot === 'A' ? _abSnapshotA : _abSnapshotB;
  if (!snap) return;
  const wrap = cachedEl('abAudioWrap');
  if (_abCurrentUrl) URL.revokeObjectURL(_abCurrentUrl);
  _abCurrentUrl = URL.createObjectURL(snap.blob);
  wrap.innerHTML = `<audio controls src="${_abCurrentUrl}"></audio>`;
}

// ── setReferenceFile ─────────────────────────────────────────
export function setReferenceFile(file, libraryId = null) {
  referenceState.file = file;
  referenceState.libraryId = libraryId;
}

export function getReferenceState() { return { ...referenceState }; }
