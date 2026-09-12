// js/file.js — upload + library
import * as state from './state.js';

export function initFileUpload(container, { onFileSelected }) {
  const btnUpload = document.getElementById('btn-upload');
  const fileInput = document.getElementById('file-input');

  if (btnUpload && fileInput) {
    btnUpload.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      state.set('selectedFile', file);
      updateTrackInfo(file);
      onFileSelected?.(file);
    });
  }

  const center = container.querySelector ? container : document.getElementById('center-console');
  if (center) {
    center.addEventListener('dragover', (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
    center.addEventListener('drop', (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      if (!file || !file.type.startsWith('audio/')) return;
      state.set('selectedFile', file);
      updateTrackInfo(file);
      onFileSelected?.(file);
    });
  }
}

function updateTrackInfo(file) {
  const nameEl = document.getElementById('info-name');
  const metaEl = document.getElementById('headerTrackMeta');
  if (nameEl) nameEl.textContent = file.name;

  const audio = new Audio();
  const objUrl = URL.createObjectURL(file);
  audio.src = objUrl;
  audio.addEventListener('loadedmetadata', () => {
    const dur = formatDuration(audio.duration);
    const size = formatSize(file.size);
    if (metaEl) metaEl.textContent = `${dur} · ${size}`;
    // No revocar acá: el audio solo usa el URL para metadata.
    // El preview/playback crea su propio blob (ver preview-controller.js).
  });
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

export function formatDuration(sec) {
  if (!Number.isFinite(sec)) return '--:--';
  const m = Math.floor(sec / 60).toString().padStart(2, '0');
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}
