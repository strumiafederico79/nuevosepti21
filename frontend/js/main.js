// js/main.js — orquesta todo
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
import * as workspaceTabs from './workspace-tabs.js';
import * as flexLayout from './flex-layout.js';
import * as headerResize from './header-resize.js';
import * as consoleResize from './console-resize.js';

// ── Init (orden correcto) ─────────────────────────────────────
initObservability();
startCleanup();
register('main-eventListeners', cleanupEventListeners);
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
  window.dispatchEvent(new CustomEvent('lgmdm:authenticated', { detail: { authenticated: isAuth } }));
});

bindOnce($('btn-logout'), 'click', () => auth.logout(), 'logout');

// ── Theme ─────────────────────────────────────────────────────
const themeSelect = $('theme-select');
const savedTheme = localStorage.getItem('lgmdm-theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);
if (themeSelect) {
  themeSelect.value = savedTheme;
  themeSelect.addEventListener('change', () => {
    const theme = themeSelect.value;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('lgmdm-theme', theme);
    record('theme-change', theme);
  });
}

// ── File Upload ───────────────────────────────────────────────
file.initFileUpload(document, {
  onFileSelected: (f) => {
    window.dispatchEvent(new CustomEvent('lgmdm:file-selected', { detail: { file: f } }));
    record('file-selected', f?.size || 0, { name: f?.name });
  },
});

// ── Master Button ─────────────────────────────────────────────
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
      onProgress: (s, p, stage) => showProgress('processing', `${stage || 'Procesando'}… ${p != null ? Math.round(p) + '%' : ''}`, p),
      onDone: (data) => {
        showProgress('done', 'Mastering completado ✓');
        $('btn-master').disabled = false;
        $('btn-download').style.display = 'block';
        $('btn-download').onclick = () => mastering.downloadMaster(jobId);
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

// ── Analyze ───────────────────────────────────────────────────
bindOnce($('btn-analyze'), 'click', async () => {
  const file_ = state.get('selectedFile');
  if (!file_) { showProgress('error'); return; }
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
