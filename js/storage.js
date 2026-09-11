// js/storage.js — localStorage wrapper con try/catch
export function get(key, fallback = null) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
export function set(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}
export function remove(key) {
  try { localStorage.removeItem(key); } catch {}
}
