// js/header-resize.js — Header height resizable
import * as storage from './storage.js';
import { makeResizable } from './make-resizable.js';

const STORAGE_KEY = 'lgmdm.headerHeight';
const MIN_H = 40;
const MAX_H = 140;

export function init() {
  if (init.done) return;
  init.done = true;

  const saved = storage.get(STORAGE_KEY, null);
  if (saved) document.documentElement.style.setProperty('--header-h', saved);

  const handle = document.getElementById('headerResizeHandle');
  if (!handle) return;

  makeResizable(handle, {
    axis: 'y',
    getSize: () => parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 48,
    setSize: (v) => document.documentElement.style.setProperty('--header-h', v + 'px'),
    min: MIN_H,
    max: MAX_H,
    onEnd: () => {
      const val = getComputedStyle(document.documentElement).getPropertyValue('--header-h');
      storage.set(STORAGE_KEY, val);
    },
  });
}
