// js/system-status.js — pinta CPU/RAM/health-dot del header en vivo.
// BUGFIX Sep 11: los spans #cpu-usage / #ram-usage ya existían en el DOM pero
// nadie los actualizaba. Backend expone /dashboard (HTTP GET auth) y /ws/dashboard
// (WS con ticket) que devuelven {cpu_percent, ram_percent, ram_used_mb,
// ram_total_mb, queue, active_job}. Preferimos WebSocket porque ya emite a 1Hz
// sin sobrecargar el loop principal; si falla (sin token, browser viejo, etc.)
// caemos a HTTP polling cada 2s.

import { apiFetch, apiBase, getToken } from './api.js';

const THRESHOLDS = { warn: 75, crit: 92 };
let ws = null;
let pollTimer = null;
let stopped = false;

function paint(stats) {
  const cpu = document.getElementById('cpu-usage');
  const ram = document.getElementById('ram-usage');
  const dot = document.getElementById('health-dot');
  if (!stats || !cpu || !ram) return;
  cpu.textContent = Number.isFinite(stats.cpu_percent) ? stats.cpu_percent.toFixed(0) : '--';
  ram.textContent = Number.isFinite(stats.ram_percent) ? stats.ram_percent.toFixed(0) : '--';
  if (dot) {
    dot.classList.remove('warn', 'crit');
    const usedMb = stats.ram_used_mb != null ? `${stats.ram_used_mb}MB` : '';
    const totMb = stats.ram_total_mb != null ? `${stats.ram_total_mb}MB` : '';
    const tip = `Backend OK · ${usedMb}${totMb ? ` / ${totMb}` : ''} · queue:${stats.queue?.queued ?? '-'} procs:${stats.queue?.processing ?? '-'}`;
    dot.title = tip;
    const maxCpu = Number(stats.cpu_percent) || 0;
    const maxRam = Number(stats.ram_percent) || 0;
    if (maxCpu >= THRESHOLDS.crit || maxRam >= THRESHOLDS.crit) dot.classList.add('crit');
    else if (maxCpu >= THRESHOLDS.warn || maxRam >= THRESHOLDS.warn) dot.classList.add('warn');
  }
}

async function startHttpPolling() {
  if (pollTimer || stopped) return;
  async function tick() {
    if (stopped) return;
    try {
      const r = await apiFetch(`${apiBase()}/dashboard`);
      if (r.ok) paint(await r.json());
    } catch (_) {}
  }
  await tick();
  pollTimer = setInterval(tick, 2000);
}

async function fetchWsTicket() {
  try {
    const r = await apiFetch(`${apiBase()}/auth/ws-ticket`);
    if (!r.ok) return null;
    const d = await r.json();
    return d?.ticket || d?.token || null;
  } catch (_) { return null; }
}

export async function startSystemStatus() {
  stopped = false;
  // Sin token no llegamos a /ws ni a /dashboard → dejamos "--".
  if (!getToken()) return;
  const ticket = await fetchWsTicket();
  if (!ticket || stopped) { await startHttpPolling(); return; }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  // apiBase() puede devolver "/api" o vacío según config; rehacemos URL completa.
  const host = location.host;
  const basePath = apiBase() || '';
  const url = `${proto}://${host}${basePath}/ws/dashboard?token=${encodeURIComponent(ticket)}`;

  try {
    ws = new WebSocket(url);
    let opened = false;
    ws.onopen = () => { opened = true; };
    ws.onmessage = (e) => { try { paint(JSON.parse(e.data)); } catch {} };
    ws.onerror = () => { /* dejar que onclose gatille fallback */ };
    ws.onclose = () => {
      ws = null;
      if (!opened && !stopped) startHttpPolling();
      else if (!stopped) startHttpPolling();
    };
    // Timebox: si en 3s no se abrió nada, cae al polling.
    setTimeout(() => { if (!opened && !stopped) { try { ws?.close(); } catch {} ; startHttpPolling(); } }, 3000);
  } catch {
    await startHttpPolling();
  }
}

export function stopSystemStatus() {
  stopped = true;
  try { ws?.close(); } catch {}
  ws = null;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}
