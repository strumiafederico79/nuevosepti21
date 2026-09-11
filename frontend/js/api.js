// js/api.js — HTTP transport
const API_BASE = 'https://masteringstudio-api.duckdns.org';

let _token = null;

export function setToken(token) { _token = token; }
export function getToken() { return _token; }

export function authHeaders() {
  const h = {};
  if (_token) h['Authorization'] = `Bearer ${_token}`;
  return h;
}

export async function apiFetch(path, options = {}) {
  const { timeout = 30000, maxRetries = 2, ...fetchOpts } = options;
  fetchOpts.headers = { ...authHeaders(), ...fetchOpts.headers };

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    fetchOpts.signal = controller.signal;

    try {
      const res = await fetch(path, fetchOpts);
      clearTimeout(timer);
      return res;
    } catch (err) {
      clearTimeout(timer);
      if (attempt === maxRetries || err.name === 'AbortError') throw err;
      await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
    }
  }
}

export function apiBase() { return API_BASE; }

export async function downloadAuthenticated(path, options = {}) {
  const { filename = 'stfx-download' } = options;
  const res = await apiFetch(path, options);
  if (!res.ok) throw new Error(`Download failed: HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  return { response: res, blob };
}
