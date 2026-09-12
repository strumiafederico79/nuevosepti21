// js/main.js — orquesta todo
self.__LGMDM_BUILD__ = '1.1';
console.log('%c[LGMDM] build 1.1', 'color:#08d;font-weight:bold;background:#000;padding:2px 6px;border-radius:3px');
import * as state from './state.js';
import * as auth from './auth.js';
import * as file from './file.js';
import * as mastering from './mastering.js';
import * as viz from './visualizers.js';
import { handleError, showToast, showProgress } from './00-error-handler.js';
import { cachedEl, bindOnce } from './00-dom-safety.js';
import { initObservability, record } from './00-observability.js';
import { cleanup, register, start as startCleanup } from './00-memory-cleanup.js';
import * as studio from './studio-controller.js';
import * as console_ from './master-console.js';
import * as reference from './reference-mastering.js';
import * as spectrum from './spectrum-controller.js';

import * as previewCtrl from './preview-controller.js';
import * as analysisView from './analysis-view.js';
import * as metersDash from './meters-dashboard.js';
import * as aiAssistant from './ai-assistant-ux.js';
import * as refLibPicker from './reference-library-picker.js';
import * as libraryPicker from './library-picker.js';
import * as systemStatus from './system-status.js';
import * as workspaceTabs from './workspace-tabs.js';
import * as flexLayout from './flex-layout.js';
import * as headerResize from './header-resize.js';
import * as consoleResize from './console-resize.js';

// ── Init (orden correcto) ─────────────────────────────────────
initObservability();
startCleanup();
register('main-eventListeners', cleanupEventListeners);
// Auto-limpieza one-shot: descarta savedValues huérfanos o '__bypassed__'
// de plugins con bypassInput (que ya no usan ese flag) y de 'limiter' que
// por bug histórico podía persistir deshabilitando el brick-wall.
const purged = studio.purgeStaleSavedValues();
if (purged > 0) console.log('%c[LGMDM] purged ' + purged + ' stale localStorage flags', 'color:#fa3');
studio.install();
console_.wire();
spectrum.init();

const $ = cachedEl;
const resultsArea = $('results-area');

// ── Auth ──────────────────────────────────────────────────────
auth.initAuth($('auth-container'));

state.subscribe('authenticated', (isAuth) => {
  $('auth-overlay').style.display = isAuth ? 'none' : 'flex';
  $('app-main').style.display = isAuth ? 'grid' : 'none';
  $('username-display').textContent = auth.getUser()?.email || '—';
  record('auth-state', isAuth ? 1 : 0);
  if (isAuth) systemStatus.startSystemStatus();
  else systemStatus.stopSystemStatus();
  window.dispatchEvent(new CustomEvent('lgmdm:authenticated', { detail: { authenticated: isAuth } }));
});

bindOnce($('btn-logout'), 'click', () => auth.logout(), 'logout');

// ── Library picker — botón 📚 junto a 📂 ─────────────────────
bindOnce($('btn-library'), 'click', () => libraryPicker.openLibrary(), 'btn-library');
window.addEventListener('keydown', (e) => {
  // Esc cierra el modal si está abierto; el binding global no compite con otros handlers.
  if (e.key === 'Escape' && document.getElementById('library-modal') &&
      !document.getElementById('library-modal').classList.contains('hidden')) {
    libraryPicker.closeLibrary();
  }
});

// ── Theme picker (5 swatches) ───────────────────────────────────
const THEMES = [
  { id: 'studio-dark',       label: 'Studio Dark' },
  { id: 'velvet-wired',      label: 'Velvet Wired' },
  { id: 'carbon-gold',       label: 'Carbon Gold' },
  { id: 'slate-arctic',      label: 'Slate Arctic' },
  { id: 'sunset-mastering',  label: 'Sunset Mastering' },
];
const LEGACY_THEME_MAP = { 'neon-cyberpunk': 'velvet-wired' };
const THEME_STORAGE_KEY = 'lgmdm.theme';

function applyTheme(themeId) {
  const def = THEMES.find(t => t.id === themeId) || THEMES[0];
  document.documentElement.setAttribute('data-theme', def.id);
  try { localStorage.setItem(THEME_STORAGE_KEY, def.id); } catch {}
  const nameEl = document.getElementById('theme-name');
  if (nameEl) nameEl.textContent = def.label;
  document.querySelectorAll('[data-theme-pick]').forEach((btn) => {
    const active = btn.dataset.themePick === def.id;
    btn.setAttribute('aria-checked', active ? 'true' : 'false');
  });
  record('theme-apply', def.id);
}

(function initThemePicker() {
  let saved;
  try { saved = localStorage.getItem(THEME_STORAGE_KEY); } catch {}
  // Migración de ids viejos → nombres nuevos (Sep 11).
  if (saved && LEGACY_THEME_MAP[saved]) {
    saved = LEGACY_THEME_MAP[saved];
    try { localStorage.setItem(THEME_STORAGE_KEY, saved); } catch {}
  }
  applyTheme(saved && THEMES.some(t => t.id === saved) ? saved : 'studio-dark');

  bindOnce(document.getElementById('theme-swatches'), 'click', (e) => {
    const btn = e.target.closest('[data-theme-pick]');
    if (!btn) return;
    applyTheme(btn.dataset.themePick);
  }, 'theme-swatches');
})();

// ── File Upload ───────────────────────────────────────────────
file.initFileUpload(document, {
  onFileSelected: (f) => {
    window.dispatchEvent(new CustomEvent('lgmdm:file-selected', { detail: { file: f } }));
    record('file-selected', f?.size || 0, { name: f?.name });
  },
});

// ── Master Button ─────────────────────────────────────────────
function _showDownloadButton(jobId, filename = 'mastered.wav') {
  // BUGFIX Sep 11: el botón quedaba con display:block pero si el cache del DOM
  // devolvía null en algún rerender, la asignación .style crasheaba silenciosa
  // y nunca aparecía nada. Ahora hacemos fallback directo a getElementById
  // y dejamos registrado currentJobId para debugging.
  const btn = document.getElementById('btn-download');
  if (!btn) { console.error('[download] #btn-download no existe en DOM'); return; }
  btn.style.display = 'inline-flex';
  btn.classList.add('btn-download-ready');
  btn.onclick = () => mastering.downloadMaster(jobId, filename);
  btn.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
}

bindOnce($('btn-master'), 'click', async () => {
  const file_ = state.get('selectedFile');
  if (!file_) { showProgress('error'); return; }
  resultsArea.innerHTML = '';
  showProgress('processing');
  $('btn-master').disabled = true;
  const t0 = performance.now();
  try {
    const { jobId } = await mastering.submitJob(file_);
    showProgress('queued', `Job ${jobId.slice(0, 8)}… en cola`);
    record('master-submit', performance.now() - t0, { fileSize: file_.size });
    mastering.pollJob(jobId, {
      onProgress: (s, p, stage) => showProgress(`${stage || 'Procesando'}… ${p != null ? Math.round(p) + '%' : ''}`, p),
      onDone: (data) => {
        showProgress('done', 'Mastering completado ✓');
        $('btn-master').disabled = false;
        try { _showDownloadButton(jobId); } catch (e) { console.error('[manual master] download button:', e); }
        record('master-complete', performance.now() - t0);
        if (data.analysis_after) viz.renderAnalysisComparison(data.analysis_before, data.analysis_after, resultsArea);
      },
      onError: (msg) => { showProgress('error', msg); $('btn-master').disabled = false; },
    });
  } catch (e) {
    handleError(e, 'Error al enviar mastering');
    $('btn-master').disabled = false;
  }
}, 'btn-master');

// ── Master with Reference ─────────────────────────────────────
bindOnce($('btn-master-ref'), 'click', async () => {
  const file_ = state.get('selectedFile');
  if (!file_) { showProgress('error'); return; }
  reference.submitReferenceMasterJob(file_);
}, 'btn-master-ref');

// ── Auto-Master (IA decide los parámetros) ─────────────────────
// BUGFIX Sep 11: el botón 🧠 Auto (#btn-auto-master) vivía en index.html pero
// nunca tuvo click handler — pressing it no disparaba ningún submit. Ahora
// envía a /ai/auto-master que hace análisis + decisión IA + cola de job.
// El ai-assistant-ux dispara este mismo handler cuando propone "Masteriza esto por mí".
bindOnce($('btn-auto-master'), 'click', async () => {
  const file_ = state.get('selectedFile');
  if (!file_) { showProgress('error'); return; }
  resultsArea.innerHTML = '';
  showProgress('processing', 'IA analiza y decide la cadena…');
  $('btn-auto-master').disabled = true;
  $('btn-master').disabled = true;
  const t0 = performance.now();
  try {
    const { jobId, aiDecision } = await mastering.submitAutoMaster(file_);
    showProgress('queued', `Job ${jobId.slice(0, 8)}… en cola`);
    record('automaster-submit', performance.now() - t0, { fileSize: file_.size });
    mastering.pollJob(jobId, {
      onProgress: (s, p, stage) => showProgress(`${stage || 'Procesando'}… ${p != null ? Math.round(p) + '%' : ''}`, p),
      onDone: (data) => {
        showProgress('done', 'Auto-mastering completado ✓');
        $('btn-auto-master').disabled = false;
        $('btn-master').disabled = false;
        try { _showDownloadButton(jobId); } catch (e) { console.error('[auto master] download button:', e); }
        record('automaster-complete', performance.now() - t0);
        if (data.analysis_after) viz.renderAnalysisComparison(data.analysis_before, data.analysis_after, resultsArea);
        if (aiDecision?.platform && typeof showToast === 'function') {
          showToast(`IA eligió target ${aiDecision.platform}`, 'info', 4000);
        }
      },
      onError: (msg) => {
        showProgress('error', msg);
        $('btn-auto-master').disabled = false;
        $('btn-master').disabled = false;
      },
    });
  } catch (e) {
    handleError(e, 'Error en auto-mastering');
    $('btn-auto-master').disabled = false;
    $('btn-master').disabled = false;
  }
}, 'btn-auto-master');

// ── Analyze ───────────────────────────────────────────────────
bindOnce($('btn-analyze'), 'click', async () => {
  const file_ = state.get('selectedFile');
  if (!file_) { showProgress('error'); return; }
  // Deshabilitar durante toda la operación asíncrona
  $('btn-analyze').disabled = true;
  resultsArea.innerHTML = '';
  try {
    showProgress('processing');
    await analysisView.requestAnalysis();
    showProgress('done', 'Análisis completo');
  } catch (e) {
    handleError(e, 'Error al analizar');
  } finally {
    $('btn-analyze').disabled = false;
  }
}, 'btn-analyze');

// ── Init new modules ──────────────────────────────────────────
previewCtrl.init();
analysisView.init();
metersDash.init();
aiAssistant.init();
refLibPicker.init();
workspaceTabs.init();
flexLayout.init();
headerResize.init();
consoleResize.init();

// ── Helpers ───────────────────────────────────────────────────
function cleanupEventListeners() {
  resultsArea.innerHTML = '';
}
