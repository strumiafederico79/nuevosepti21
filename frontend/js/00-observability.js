// js/00-observability.js — Métricas, performance observers, error capture
import { apiFetch } from './api.js';

const _metrics = Object.create(null);
const _observers = [];
let _initialized = false;
const _observersSupported = typeof PerformanceObserver !== 'undefined';

function endpoint() {
  const node = document.querySelector('meta[name="lgmdm-observability-endpoint"]');
  return node?.content?.trim() || '';
}

export function record(name, value, extra = {}) {
  _metrics[name] = { value, at: Date.now(), ...extra };
  window.dispatchEvent(new CustomEvent('lgmdm:metric', { detail: _metrics[name] }));
}

export function reportToBackend(name, value, extra = {}) {
  const url = endpoint();
  if (!url) return;
  apiFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ metric: name, value, ...extra, path: location.pathname }),
  }).catch(() => {});
}

export function captureError(error, context = {}) {
  const err = error instanceof Error ? error : new Error(String(error));
  const payload = { message: err.message, stack: err.stack, ...context, at: new Date().toISOString(), path: location.pathname };
  const url = endpoint();
  if (url) {
    apiFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: payload }),
    }).catch(() => {});
  } else {
    console.error('[Observability]', err, context);
  }
}

function observePerformance() {
  if (!_observersSupported) return;

  try {
    const paint = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'first-contentful-paint') {
          record('FCP', entry.startTime);
          reportToBackend('FCP', entry.startTime);
        }
      }
    });
    paint.observe({ type: 'paint', buffered: true });
    _observers.push(paint);
  } catch (_) {}

  try {
    const lcp = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const last = entries[entries.length - 1];
      if (last) {
        record('LCP', last.startTime);
        reportToBackend('LCP', last.startTime);
      }
    });
    lcp.observe({ type: 'largest-contentful-paint', buffered: true });
    _observers.push(lcp);
  } catch (_) {}

  try {
    let cls = 0;
    const clsObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!entry.hadRecentInput) cls += entry.value;
      }
      record('CLS', cls);
    });
    clsObserver.observe({ type: 'layout-shift', buffered: true });
    _observers.push(clsObserver);
  } catch (_) {}

  try {
    const inpObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.interactionId) {
          record('INP.last', entry.duration, { interactionId: entry.interactionId });
        }
      }
    });
    inpObserver.observe({ type: 'event', buffered: true, durationThreshold: 40 });
    _observers.push(inpObserver);
  } catch (_) {}
}

function recordNavigation() {
  try {
    const nav = performance.getEntriesByType?.('navigation')?.[0];
    if (nav && Number.isFinite(nav.responseStart)) {
      record('TTFB', nav.responseStart - nav.requestStart);
    }
  } catch (_) {}
}

export function initObservability() {
  if (_initialized) return;
  _initialized = true;
  observePerformance();
  recordNavigation();
  window.addEventListener('error', (e) => captureError(e.error || new Error(e.message), { source: e.filename, line: e.lineno }));
  window.addEventListener('unhandledrejection', (e) => captureError(e.reason || new Error('Unhandled rejection')));
}

export function getMetrics() { return { ..._metrics }; }
