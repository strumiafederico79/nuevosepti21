// js/spectrum-controller.js — Spectrum real-time with animated bars
const state = { last: null, consoleCanvasId: 'spectrum-canvas', dirty: false, raf: 0 };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

export function drawCanvas(canvas, bands, minDb = -80, maxDb = 0) {
  if (!canvas || !Array.isArray(bands) || !bands.length) return;
  const r = canvas.getBoundingClientRect(); const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(260, Math.floor((r.width || 600) * dpr));
  const h = Math.max(120, Math.floor((r.height || 180) * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d'); if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, w, h);
  const data = bands.map(Number).slice(0, 64); const count = data.length;
  const gap = Math.max(1.5, 2 * dpr); const bw = Math.max(2, (w - gap * (count - 1)) / count);
  for (let i = 0; i < count; i++) {
    const db = clamp(Number.isFinite(data[i]) ? data[i] : minDb, minDb, maxDb);
    const t = (db - minDb) / (maxDb - minDb); const bh = Math.max(2 * dpr, t * h * .88);
    const x = i * (bw + gap), y = h - bh;
    const g = ctx.createLinearGradient(0, h, 0, 0);
    g.addColorStop(0, 'rgba(87,230,255,.18)'); g.addColorStop(.5, 'rgba(87,230,255,.72)');
    g.addColorStop(.82, 'rgba(169,140,255,.9)'); g.addColorStop(1, 'rgba(255,202,101,1)');
    ctx.fillStyle = g; ctx.fillRect(x, y, bw, bh);
  }
  const refY = h * ((-18 - minDb) / (maxDb - minDb));
  ctx.strokeStyle = 'rgba(255,255,255,.12)'; ctx.lineWidth = dpr; ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.beginPath(); ctx.moveTo(0, refY); ctx.lineTo(w, refY); ctx.stroke(); ctx.setLineDash([]);
}

export function normalizeBands(input) {
  if (Array.isArray(input)) return input.slice();
  if (input && typeof input === 'object') {
    const bands = input.bands_db || input.magnitudes_db || input.values_db || input.values;
    return Array.isArray(bands) ? bands.slice() : null;
  }
  return null;
}

export function setData(bands) { state.last = normalizeBands(bands); state.dirty = true; schedule(); }
export function clear() { state.last = null; state.dirty = false; schedule(); }

function draw() {
  state.raf = 0;
  if (!state.last) return;
  const canvas = document.getElementById(state.consoleCanvasId);
  drawCanvas(canvas, state.last);
  state.dirty = false;
}

function schedule() { if (!state.raf) state.raf = requestAnimationFrame(draw); }
export function redraw() { schedule(); }

export function init() {
  if (init.done) return; init.done = true;
  window.addEventListener('resize', schedule, { passive: true });
  window.addEventListener('lgmdm:metrics', (e) => {
    const sp = e.detail?.metrics?.spectrum || e.detail?.spectrum;
    if (sp != null) setData(sp);
  });
}
