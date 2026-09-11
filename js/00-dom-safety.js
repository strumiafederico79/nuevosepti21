// js/00-dom-safety.js — Acceso seguro al DOM + bindOnce
const _domCache = new Map();
const _bindOnceKeys = new WeakMap();

export function cachedEl(id) {
  if (_domCache.has(id)) {
    const el = _domCache.get(id);
    if (el && document.body.contains(el)) return el;
    _domCache.delete(id);
  }
  const el = typeof id === 'string' ? document.getElementById(id) : id;
  if (el) _domCache.set(id, el);
  return el;
}

export function bindOnce(el, type, fn, key, opts) {
  if (!el || typeof fn !== 'function') return false;
  const keyMap = _bindOnceKeys.get(el) || (_bindOnceKeys.set(el, {}), _bindOnceKeys.get(el));
  const k = key || `${type}:${fn.name || 'anon'}`;
  if (keyMap[k]) return false;
  el.addEventListener(type, fn, opts);
  keyMap[k] = true;
  return true;
}

export function escapeHtml(value) {
  if (value == null) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
