// js/state.js — reactive store
const _store = {};
const _listeners = {};

export function get(key) {
  return _store[key];
}

export function set(key, value) {
  const prev = _store[key];
  _store[key] = value;
  if (_listeners[key]) {
    _listeners[key].forEach(fn => {
      try { fn(value, prev); } catch (e) { console.error(`[state] listener error for "${key}":`, e); }
    });
  }
}

export function subscribe(key, callback) {
  if (!_listeners[key]) _listeners[key] = [];
  _listeners[key].push(callback);
  return () => {
    _listeners[key] = _listeners[key].filter(fn => fn !== callback);
  };
}
