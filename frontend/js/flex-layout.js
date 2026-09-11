// js/flex-layout.js — Layout 3 columnas resizable
import * as storage from './storage.js';
import { makeResizable } from './make-resizable.js';

const STORAGE_KEY_W = 'lgmdm:flex-sidebar-w';
const STORAGE_KEY_COLLAPSED = 'lgmdm:flex-sidebar-collapsed';
const MIN_LEFT = 180;
const MAX_LEFT_PCT = 0.42;
const MIN_RIGHT = 200;
const MAX_RIGHT_PCT = 0.42;

let destroyLeft, destroyRight;

function applyLayout() {
  const root = document.documentElement;
  const savedW = storage.get(STORAGE_KEY_W, null);
  const vw = window.innerWidth;
  const maxLeft = Math.floor(vw * MAX_LEFT_PCT);
  const leftW = savedW != null ? Math.min(Math.max(savedW, MIN_LEFT), maxLeft) : 220;
  root.style.setProperty('--left-w', leftW + 'px');
  return leftW;
}

function saveWidth() {
  const val = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--left-w')) || 220;
  storage.set(STORAGE_KEY_W, val);
}

function toggleCollapse(persist = true) {
  const sidebar = document.getElementById('left-panel');
  const handle = document.getElementById('leftResizeHandle');
  if (!sidebar) return;
  const collapsed = sidebar.classList.toggle('collapsed');
  if (collapsed) {
    sidebar.style.display = 'none';
    if (handle) handle.style.display = 'none';
    document.documentElement.style.setProperty('--left-w', '0px');
  } else {
    sidebar.style.display = '';
    if (handle) handle.style.display = '';
    applyLayout();
  }
  if (persist) storage.set(STORAGE_KEY_COLLAPSED, collapsed);
  window.dispatchEvent(new CustomEvent('lgmdm:sidebar-toggle', { detail: { collapsed } }));
}

export function init() {
  if (init.done) return;
  init.done = true;

  applyLayout();

  const collapsed = storage.get(STORAGE_KEY_COLLAPSED, false);
  if (collapsed) toggleCollapse(false);

  const leftHandle = document.getElementById('leftResizeHandle');
  const rightHandle = document.getElementById('rightResizeHandle');

  if (leftHandle) {
    destroyLeft = makeResizable(leftHandle, {
      axis: 'x',
      getSize: () => parseInt(getComputedStyle(document.documentElement).getPropertyValue('--left-w')) || 220,
      setSize: (v) => document.documentElement.style.setProperty('--left-w', v + 'px'),
      min: MIN_LEFT,
      max: () => Math.floor(window.innerWidth * MAX_LEFT_PCT),
      onEnd: saveWidth,
    });
  }

  if (rightHandle) {
    destroyRight = makeResizable(rightHandle, {
      axis: 'x',
      invert: true,
      getSize: () => parseInt(getComputedStyle(document.documentElement).getPropertyValue('--right-w')) || 280,
      setSize: (v) => document.documentElement.style.setProperty('--right-w', v + 'px'),
      min: MIN_RIGHT,
      max: () => Math.floor(window.innerWidth * MAX_RIGHT_PCT),
      onEnd: () => storage.set('lgmdm:flex-right-w', parseInt(getComputedStyle(document.documentElement).getPropertyValue('--right-w')) || 280),
    });
  }

  window.addEventListener('resize', () => applyLayout(), { passive: true });
}
