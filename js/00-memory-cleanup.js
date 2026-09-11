// js/00-memory-cleanup.js — Registro y ejecución de cleanup callbacks
const _cleanupFns = new Map();
let _running = false;

export function register(name, fn) {
  if (typeof fn !== 'function') return false;
  _cleanupFns.set(name, fn);
  return true;
}

export function cleanup() {
  _cleanupFns.forEach((fn, name) => {
    try { fn(); } catch (e) { console.warn(`[Cleanup] Error en "${name}":`, e); }
  });
}

export function start() {
  if (_running) return;
  _running = true;
  window.addEventListener('beforeunload', cleanup);
}

export function stop() {
  if (!_running) return;
  _running = false;
  window.removeEventListener('beforeunload', cleanup);
}
