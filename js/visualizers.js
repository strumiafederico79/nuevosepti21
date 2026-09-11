// js/visualizers.js — render analysis results
import { escapeHtml } from './00-dom-safety.js';

export function renderAnalysisSingle(data, container) {
  if (!data || typeof data !== 'object') return;
  const grid = document.createElement('div');
  grid.className = 'analysis-grid';
  grid.innerHTML = `
    <div class="analysis-panel"><h3>Métricas del audio</h3>${metricsHtml(data, null)}</div>
    ${perceptualPanelHtml(data)}
  `;
  container.appendChild(grid);
  renderProfessionalMeter(data, container);
}

export function renderAnalysisComparison(before, after, container) {
  const grid = document.createElement('div');
  grid.className = 'analysis-grid';
  grid.innerHTML = `
    <div class="analysis-panel"><h3>Antes</h3>${metricsHtml(before, null)}</div>
    <div class="analysis-panel"><h3>Después</h3>${metricsHtml(after, before)}</div>
    ${perceptualPanelHtml(after, '— Después')}
  `;
  container.appendChild(grid);
  renderProfessionalMeter(after, container);
}

export function renderPerceptual(analysis, container, subtitle) {
  if (!analysis?.perceptual) return;
  const grid = document.createElement('div');
  grid.className = 'analysis-grid';
  grid.innerHTML = perceptualPanelHtml(analysis, subtitle);
  container.appendChild(grid);
}

export function renderAdvicePanel(adviceData, title, container, subtitle) {
  const panel = document.createElement('div');
  panel.className = 'advice-panel';
  const score = adviceData.score ?? 0;
  const grade = adviceData.grade ?? '';
  const issues = adviceData.issues ?? [];
  const tips = adviceData.tips ?? [];
  const gradeClass = grade === 'Excelente' ? 'grade-ex' : grade === 'Buena' ? 'grade-good' : grade === 'Aceptable' ? 'grade-ok' : 'grade-bad';
  const issuesHtml = issues.length ? `<ul class="advice-issues">${issues.map(i => `<li>${escapeHtml(i)}</li>`).join('')}</ul>` : '';
  const tipsHtml = tips.length ? `<ul class="advice-tips">${tips.map(t => `<li>${escapeHtml(t)}</li>`).join('')}</ul>` : '';
  panel.innerHTML = `
    <h3>${escapeHtml(title)}${subtitle ? ` <span>${escapeHtml(subtitle)}</span>` : ''}</h3>
    <div class="advice-score-row">
      <div class="advice-score-circle"><span class="score-num">${score}</span><span class="score-label">/ 100</span></div>
      <div><div class="advice-grade ${gradeClass}">${grade}</div><div>${issues.length} problema${issues.length !== 1 ? 's' : ''}</div></div>
    </div>
    ${issuesHtml}${tipsHtml}
  `;
  container.appendChild(panel);
}

export function renderFFT(series, container) {
  const wrap = document.createElement('div');
  wrap.className = 'fft-wrap';
  const legendHtml = series.map(s =>
    `<span style="color:${s.color || 'var(--accent)'}">■</span> <span>${escapeHtml(s.label)}</span>`
  ).join('');
  wrap.innerHTML = `<h3>Spectrum Analyzer (FFT)</h3><canvas></canvas><div class="fft-legend">${legendHtml}</div>`;
  container.appendChild(wrap);
  drawFFTOnCanvas(wrap.querySelector('canvas'), series);
}

export function renderLoudnessMeter(lufsValue, container) {
  const wrap = document.createElement('div');
  wrap.className = 'loudness-meter';
  const pct = Math.max(0, Math.min(100, ((lufsValue + 40) / 40) * 100));
  const color = lufsValue > -6 ? 'var(--red)' : lufsValue > -9 ? 'var(--yellow)' : 'var(--green)';
  wrap.innerHTML = `
    <h3>Loudness Meter (LUFS)</h3>
    <div class="lufs-display">
      <span class="lufs-number">${lufsValue.toFixed(1)}</span>
      <div class="lufs-bar-track"><div class="lufs-bar-fill" style="width:${pct}%;background:${color}"></div></div>
    </div>
    <div class="lufs-zones"><span>-40</span><span>-24</span><span>-18</span><span>-14</span><span>-9</span><span>-6</span><span>0</span></div>
    <div class="lufs-target">Target Spotify/YouTube: -14 LUFS · Club: -9 LUFS</div>
  `;
  container.appendChild(wrap);
}

export function renderProfessionalMeter(a, container) {
  if (!a) return;
  const wrap = document.createElement('div');
  wrap.className = 'professional-meter';
  const rows = [
    { label: 'True Peak', value: a.true_peak_db, unit: 'dBTP', status: a.true_peak_db > -0.5 ? 'bad' : a.true_peak_db > -1.2 ? 'warn' : 'good' },
    { label: 'PLR', value: a.plr_db, unit: 'dB', status: a.plr_db > 10 ? 'good' : a.plr_db > 6 ? 'warn' : 'bad' },
    { label: 'Dinámica', value: a.dynamic_range_db, unit: 'dB', status: a.dynamic_range_db >= 10 ? 'good' : a.dynamic_range_db >= 6 ? 'warn' : 'bad' },
    { label: 'Correlación estéreo', value: a.stereo_correlation, unit: '', status: a.stereo_correlation < 0.85 ? 'warn' : 'good' },
    { label: 'Loudness', value: a.lufs, unit: 'LUFS', status: a.lufs >= -14 && a.lufs <= -9 ? 'good' : a.lufs >= -18 ? 'warn' : 'bad' },
  ];
  const cards = rows.map(item => {
    const suffix = item.unit ? ` ${item.unit}` : '';
    return `<div class="pm-card"><strong>${item.label}</strong><span class="metric-value ${item.status}">${item.value != null ? item.value.toFixed(1) + suffix : '—'}</span></div>`;
  }).join('');
  wrap.innerHTML = `<h3>Professional Metering</h3><div class="pm-grid">${cards}</div>`;
  container.appendChild(wrap);
}

// ── Helpers ──

function metricsHtml(a, b) {
  const rows = [
    ['LUFS', a.lufs, b?.lufs, v => `${v} LUFS`, v => v >= -14 && v <= -8 ? 'good' : v >= -18 ? 'warn' : 'bad'],
    ['RMS', a.rms_db, b?.rms_db, v => `${v} dB`, () => 'neutral'],
    ['Peak', a.peak_db, b?.peak_db, v => `${v} dBFS`, v => v > -0.5 ? 'warn' : 'good'],
    ['Rango dinámico', a.dynamic_range_db, b?.dynamic_range_db, v => `${v} dB`, v => v < 6 ? 'bad' : v <= 12 ? 'good' : 'warn'],
    ['BPM', a.bpm, b?.bpm, v => `${v}`, () => 'neutral'],
    ['Duración', a.duration_sec, null, v => `${v} s`, () => 'neutral'],
    ['Sample rate', a.sample_rate, null, v => `${v} Hz`, () => 'neutral'],
    ['Canales', a.channels, null, v => v === 1 ? 'Mono' : 'Estéreo', () => 'neutral'],
  ];
  return rows.map(([label, va, vb, fmt, cls]) =>
    `<div class="metric-row"><span class="metric-label">${label}</span><span class="metric-value ${cls(va)}">${va != null ? fmt(va) : '—'}${b && vb != null ? ` <span class="delta">${(vb - va) > 0 ? '+' : ''}${(vb - va).toFixed(1)}</span>` : ''}</span></div>`
  ).join('');
}

const PERCEPTUAL_LABELS = {
  clarity: 'Claridad', dynamic_feel: 'Dinámica', tonal_balance: 'Balance tonal',
  stereo_coherence: 'Coherencia estéreo', instrumental_definition: 'Definición',
  presence_feel: 'Presencia', mix_cohesion: 'Cohesión de mezcla',
};

function perceptualPanelHtml(a, subtitle) {
  if (!a?.perceptual) return '';
  const rows = Object.entries(PERCEPTUAL_LABELS).map(([key, label]) => {
    const val = a.perceptual[key];
    if (val == null) return '';
    const pct = Math.round(val * 100);
    return `<div class="perc-row"><span class="perc-label">${label}</span><div class="perc-bar-wrap"><div class="perc-bar" style="width:${pct}%"></div></div><span class="perc-val">${pct}%</span></div>`;
  }).join('');
  return `<div class="analysis-panel perceptual-panel"><h3>Análisis perceptual${subtitle ? ` <span>${escapeHtml(subtitle)}</span>` : ''}</h3>${rows}</div>`;
}

function drawFFTOnCanvas(canvas, series) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600;
  const H = 100;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#1a1a24';
  ctx.fillRect(0, 0, W, H);

  series.forEach((s) => {
    const data = s.data;
    if (!data || !Array.isArray(data)) return;
    const color = s.color || '#7c5cff';
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const step = W / data.length;
    data.forEach((v, i) => {
      const y = H - ((v + 80) / 80) * H;
      if (i === 0) ctx.moveTo(0, Math.max(0, Math.min(H, y)));
      else ctx.lineTo(i * step, Math.max(0, Math.min(H, y)));
    });
    ctx.stroke();
  });
}
