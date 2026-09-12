// js/mastering.js — submit job → poll → download
import { apiFetch, apiBase, downloadAuthenticated } from './api.js';
import { collectParams, buildQueryString } from './params.js';
import { getChainOverrides } from './master-console.js';
import * as state from './state.js';

export async function submitJob(file, overrides = null) {
  const fd = new FormData();
  fd.append('file', file);
  const params = overrides || { ...collectParams(), ...getChainOverrides() };
  const qs = buildQueryString(params);
  const res = await apiFetch(`${apiBase()}/master?${qs}`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const data = await res.json();
  state.set('currentJobId', data.job_id);
  return { jobId: data.job_id };
}

export async function submitSync(file, overrides = null) {
  const fd = new FormData();
  fd.append('file', file);
  const params = overrides || { ...collectParams(), ...getChainOverrides() };
  const qs = buildQueryString(params);
  const res = await apiFetch(`${apiBase()}/master/sync?${qs}`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.blob();
}

export function pollJob(jobId, callbacks, intervalMs = 1500) {
  const { onProgress, onDone, onError } = callbacks;
  const id = setInterval(async () => {
    try {
      const res = await apiFetch(`${apiBase()}/job/${jobId}`);
      const data = await res.json();
      if (data.status === 'queued' || data.status === 'processing') {
        onProgress?.(data.status, data.progress, data.stage);
      } else if (data.status === 'done') {
        clearInterval(id);
        onDone?.(data);
      } else if (data.status === 'error') {
        clearInterval(id);
        onError?.(data.error || 'Job failed');
      }
    } catch (e) {
      clearInterval(id);
      onError?.(e.message);
    }
  }, intervalMs);
  return () => clearInterval(id);
}

export async function downloadMaster(jobId, filename = 'mastered.wav') {
  await downloadAuthenticated(`${apiBase()}/download/${jobId}`, { filename });
}

export async function downloadReport(jobId) {
  await downloadAuthenticated(`${apiBase()}/report/${jobId}`, { filename: 'report.json' });
}

export async function submitAutoMaster(file, { outputFormat = 'wav', bitDepth = 24 } = {}) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('output_format', outputFormat);
  const qs = `output_format=${encodeURIComponent(outputFormat)}&output_bit_depth=${bitDepth}`;
  const res = await apiFetch(`${apiBase()}/ai/auto-master?${qs}`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const data = await res.json();
  state.set('currentJobId', data.job_id);
  return { jobId: data.job_id, aiDecision: data.ai_decision, analysis: data.analysis };
}
