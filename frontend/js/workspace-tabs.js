// js/workspace-tabs.js — Tabs Console/Analysis/Presets
import * as storage from './storage.js';

const STORAGE_KEY = 'lgmdm.workspace';
let current = storage.get(STORAGE_KEY, 'console');

export function getWorkspace() { return current; }

export function setWorkspace(name, persist = true) {
  if (!name || name === current) return;
  const prev = current;
  current = name;
  if (persist) storage.set(STORAGE_KEY, name);

  document.body.dataset.workspace = name;

  document.querySelectorAll('.ws-tab').forEach(tab => {
    const active = tab.dataset.workspace === name;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', active);
    tab.tabIndex = active ? 0 : -1;
  });

  document.querySelectorAll('.ws-pane').forEach(pane => {
    pane.classList.toggle('active', pane.dataset.workspace === name);
  });

  window.dispatchEvent(new CustomEvent('lgmdm:workspace-change', { detail: { from: prev, to: name } }));
}

export function init() {
  if (init.done) return;
  init.done = true;

  document.querySelectorAll('.ws-tab').forEach(tab => {
    tab.addEventListener('click', () => setWorkspace(tab.dataset.workspace));
    tab.addEventListener('keydown', (e) => {
      const tabs = [...document.querySelectorAll('.ws-tab')];
      const idx = tabs.indexOf(tab);
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        tabs[(idx + 1) % tabs.length]?.focus();
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        tabs[(idx - 1 + tabs.length) % tabs.length]?.focus();
      }
    });
  });

  setWorkspace(current, false);
}
