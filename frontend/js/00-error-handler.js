// js/00-error-handler.js — Manejo centralizado de errores
let _toastEl = null;

function ensureToastContainer() {
  if (_toastEl && document.body.contains(_toastEl)) return _toastEl;
  _toastEl = document.createElement('div');
  _toastEl.id = 'toast-container';
  _toastEl.style.cssText = 'position:fixed;top:12px;right:12px;z-index:10000;display:flex;flex-direction:column;gap:8px;pointer-events:none;';
  document.body.appendChild(_toastEl);
  return _toastEl;
}

export function normalizeError(err, fallback = 'Ocurrió un error inesperado.') {
  if (err instanceof Error) return err.message || fallback;
  if (typeof err === 'string') return err;
  if (err?.message) return err.message;
  if (err?.error) return typeof err.error === 'string' ? err.error : err.error?.message || fallback;
  return fallback;
}

function classifyError(err) {
  const msg = normalizeError(err, '').toLowerCase();
  if (msg.includes('network') || msg.includes('fetch') || msg.includes('econnrefused') || msg.includes('failed to fetch')) return 'network';
  if (msg.includes('401') || msg.includes('403') || msg.includes('unauthorized') || msg.includes('forbidden') || msg.includes('token')) return 'auth';
  if (msg.includes('timeout') || msg.includes('abort')) return 'timeout';
  if (msg.includes('cors') || msg.includes('cross-origin')) return 'cors';
  if (msg.includes('decode') || msg.includes('parse') || msg.includes('json')) return 'data';
  return 'unknown';
}

function userMessage(err, fallback = 'Ocurrió un error inesperado.') {
  const type = classifyError(err);
  const map = {
    network: 'Error de conexión. Verificá tu red.',
    auth: 'Sesión expirada. Volvé a iniciar sesión.',
    timeout: 'La operación tardó demasiado. Intentá de nuevo.',
    cors: 'Error de permisos. Actualizá la página.',
    data: 'Error procesando datos del servidor.',
    unknown: fallback,
  };
  return map[type] || fallback;
}

export function showToast(message, type = 'error', duration = 4000) {
  const container = ensureToastContainer();
  const toast = document.createElement('div');
  const colors = { error: '#ef4444', warning: '#f59e0b', info: '#3b82f6', success: '#22c55e' };
  toast.style.cssText = `pointer-events:auto;padding:10px 16px;border-radius:6px;color:#fff;font-size:13px;font-family:system-ui;max-width:340px;box-shadow:0 4px 12px rgba(0,0,0,.3);opacity:0;transform:translateX(20px);transition:all .25s;cursor:pointer;background:${colors[type] || colors.error};`;
  toast.textContent = message;
  toast.addEventListener('click', () => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 250); });
  container.appendChild(toast);
  requestAnimationFrame(() => { toast.style.opacity = '1'; toast.style.transform = 'translateX(0)'; });
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 250); }, duration);
}

export function handleError(err, fallback = 'Ocurrió un error inesperado.', options = {}) {
  const { silent = false, rethrow = false, context = {}, toast = true } = options;
  const type = classifyError(err);
  if (!silent && toast) showToast(userMessage(err, fallback), type === 'auth' ? 'warning' : 'error');
  if (rethrow) throw err;
  return { message: normalizeError(err, fallback), type };
}

export function showProgress(message, totalOrText = null, pct = null) {
  const stage = document.getElementById('progress-stage');
  const fill = document.getElementById('progress-bar-fill');
  if (stage) stage.textContent = message;
  if (fill) {
    const p = pct != null ? Number(pct) : Number(totalOrText);
    if (Number.isFinite(p)) fill.style.width = Math.round(p) + '%';
    else if (message === 'done') fill.style.width = '100%';
    else if (message === 'error') fill.style.width = '0%';
  }
}
