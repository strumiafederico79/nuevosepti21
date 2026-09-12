// js/console-resize.js — Resize handles for center console elements
import * as storage from './storage.js';
import { makeResizable } from './make-resizable.js';

const KEYS = {
  spectrum: 'lgmdm.console.spectrum-h',
  waveform: 'lgmdm.console.waveform-h',
  meters:   'lgmdm.console.meters-h',
};
const DEFAULTS = { spectrum: 120, waveform: 120, meters: null };
const MIN = { spectrum: 60, waveform: 60, meters: 80 };
const MAX = { spectrum: 400, waveform: 400, meters: 500 };

function applyVar(name, val) {
  document.documentElement.style.setProperty(`--${name}`, val != null ? val + 'px' : 'auto');
}

function getVar(name) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`);
  if (!v || v.trim() === 'auto') return null;
  return parseInt(v);
}

export function init() {
  if (init.done) return;
  init.done = true;

  for (const [key, defaultVal] of Object.entries(DEFAULTS)) {
    const saved = storage.get(KEYS[key], null);
    applyVar(key + '-h', saved ?? defaultVal);
  }

  const spectrumHandle = document.getElementById('spectrumResizeHandle');
  if (spectrumHandle) {
    makeResizable(spectrumHandle, {
      axis: 'y',
      getSize: () => getVar('spectrum-h') ?? 120,
      setSize: (v) => applyVar('spectrum-h', v),
      min: MIN.spectrum,
      max: MAX.spectrum,
      onEnd: () => storage.set(KEYS.spectrum, getVar('spectrum-h')),
    });
  }

  const waveformHandle = document.getElementById('waveformResizeHandle');
  if (waveformHandle) {
    makeResizable(waveformHandle, {
      axis: 'y',
      getSize: () => getVar('waveform-h') ?? 120,
      setSize: (v) => applyVar('waveform-h', v),
      min: MIN.waveform,
      max: MAX.waveform,
      onEnd: () => storage.set(KEYS.waveform, getVar('waveform-h')),
    });
  }

  const metersHandle = document.getElementById('metersResizeHandle');
  if (metersHandle) {
    makeResizable(metersHandle, {
      axis: 'y',
      invert: true,
      getSize: () => getVar('meters-h') ?? 200,
      setSize: (v) => applyVar('meters-h', v),
      min: MIN.meters,
      max: MAX.meters,
      onEnd: () => storage.set(KEYS.meters, getVar('meters-h')),
    });
  }
}
