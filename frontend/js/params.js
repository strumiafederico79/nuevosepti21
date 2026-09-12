// js/params.js — collect params + build query string

const PARAM_DEFS = {
  input_gain_db:     { id: 's-ingain',        type: 'number',  default: '0' },
  target_peak:       { id: 's-peak',          type: 'number',  default: '-1.0' },
  use_lufs_normalize:{ id: 's-uselufs',       type: 'check',   default: false },
  target_lufs:       { id: 's-lufstarget',    type: 'number',  default: '-14' },
  comp_threshold_db: { id: 's-thresh',        type: 'number',  default: '-20' },
  comp_ratio:        { id: 's-ratio',         type: 'number',  default: '4' },
  comp_attack_ms:    { id: 's-cattack',       type: 'number',  default: '10' },
  comp_release_ms:   { id: 's-crelease',      type: 'number',  default: '100' },
  comp_makeup_db:    { id: 's-cmakeup',       type: 'number',  default: '0' },
  comp_bypass:       { id: 'comp-bypass',     type: 'bool-value',   default: true },
  oversample_mode:   { id: 's-oversample',    type: 'value',   default: '2x' },
  glue_bypass:       { id: 's-glue-bypass',   type: 'bool-value',   default: true },
  glue_threshold_db: { id: 's-glue-thresh',   type: 'number',  default: '-18' },
  glue_ratio:        { id: 's-glue-ratio',    type: 'number',  default: '2' },
  glue_attack_ms:    { id: 's-glue-attack',   type: 'number',  default: '1' },
  glue_release_ms:   { id: 's-glue-release',  type: 'number',  default: '100' },
  glue_makeup_db:    { id: 's-glue-makeup',   type: 'number',  default: '0' },
  clipper_bypass:    { id: 's-clip-bypass',   type: 'bool-value',   default: false },
  mb_bypass:         { id: 'mb-bypass',       type: 'bool-value',   default: true },
  ms_comp_bypass:    { id: 's-mscomp-bypass', type: 'bool-value',   default: true },
  mb_stereo_bypass:  { id: 'mb-stereo-bypass', type: 'bool-value',  default: false },
  clipper_ceiling:   { id: 's-clip-ceiling',  type: 'number',  default: '-0.3' },
  clipper_drive_db:  { id: 's-clip-drive',    type: 'number',  default: '0' },
  hp_cutoff:         { id: 's-hp',            type: 'number',  default: '20' },
  high_shelf_gain_db:{ id: 's-air',           type: 'number',  default: '0' },
  high_shelf_freq_hz:{ id: 's-shelf-freq',    type: 'number',  default: '10000' },
  low_shelf_gain_db: { id: 's-lowshelf',      type: 'number',  default: '0' },
  low_shelf_freq_hz: { id: 's-lowshelf-freq', type: 'number',  default: '200' },
  transient_attack:  { id: 's-tatt',          type: 'number',  default: '0' },
  transient_sustain: { id: 's-tsus',          type: 'number',  default: '0' },
  saturation_drive:  { id: 's-satdrive',      type: 'number',  default: '0' },
  saturation_mode:   { id: 's-satmode',       type: 'value',   default: 'warm' },
  saturation_mix:    { id: 's-satmix',        type: 'number',  default: '0' },
  mid_gain_db:       { id: 's-mgain',         type: 'number',  default: '0' },
  side_gain_db:      { id: 's-sgain',         type: 'number',  default: '0' },
  stereo_width_amount:{ id: 's-width',        type: 'number',  default: '100' },
  mb_stereo_low_width:  { id: 's-mb-sw-low',  type: 'number',  default: '1' },
  mb_stereo_mid_width:  { id: 's-mb-sw-mid',  type: 'number',  default: '1' },
  mb_stereo_high_width: { id: 's-mb-sw-high', type: 'number',  default: '1' },
  limiter_ceiling:   { id: 's-ceiling',       type: 'number',  default: '-1.0' },
  limiter_release_ms:{ id: 's-lrelease',      type: 'number',  default: '100' },
  preview_start_sec: { id: 's-preview-from',  type: 'number',  default: '15' },
  output_format:     { id: 's-format',        type: 'value',   default: 'wav' },
  output_bit_depth:  { id: 's-bitdepth',      type: 'value',   default: '24' },
  // EQ bands
  eq1_freq: { id: 's-eq1freq', type: 'number', default: '100' },
  eq1_gain: { id: 's-eq1gain', type: 'number', default: '0' },
  eq1_q:    { id: 's-eq1q',    type: 'number', default: '1' },
  eq2_freq: { id: 's-eq2freq', type: 'number', default: '400' },
  eq2_gain: { id: 's-eq2gain', type: 'number', default: '0' },
  eq2_q:    { id: 's-eq2q',    type: 'number', default: '1' },
  eq3_freq: { id: 's-eq3freq', type: 'number', default: '1000' },
  eq3_gain: { id: 's-eq3gain', type: 'number', default: '0' },
  eq3_q:    { id: 's-eq3q',    type: 'number', default: '1' },
  eq4_freq: { id: 's-eq4freq', type: 'number', default: '3000' },
  eq4_gain: { id: 's-eq4gain', type: 'number', default: '0' },
  eq4_q:    { id: 's-eq4q',    type: 'number', default: '1' },
  eq5_freq: { id: 's-eq5freq', type: 'number', default: '6000' },
  eq5_gain: { id: 's-eq5gain', type: 'number', default: '0' },
  eq5_q:    { id: 's-eq5q',    type: 'number', default: '1' },
  eq6_freq: { id: 's-eq6freq', type: 'number', default: '12000' },
  eq6_gain: { id: 's-eq6gain', type: 'number', default: '0' },
  eq6_q:    { id: 's-eq6q',    type: 'number', default: '1' },
  // Dynamic EQ
  dyneq_bypass:       { id: 's-dyneq-bypass',   type: 'bool-value', default: true },
  dyneq_freq:         { id: 's-dyneq-freq',     type: 'number', default: '1000' },
  dyneq_q:            { id: 's-dyneq-q',        type: 'number', default: '1' },
  dyneq_threshold_db: { id: 's-dyneq-thresh',   type: 'number', default: '-20' },
  dyneq_ratio:        { id: 's-dyneq-ratio',    type: 'number', default: '4' },
  dyneq_attack_ms:    { id: 's-dyneq-attack',   type: 'number', default: '10' },
  dyneq_release_ms:   { id: 's-dyneq-release',  type: 'number', default: '100' },
  // M/S Compressor
  ms_comp_mid_threshold_db:  { id: 's-mscomp-mid-thresh',   type: 'number', default: '-20' },
  ms_comp_mid_ratio:         { id: 's-mscomp-mid-ratio',    type: 'number', default: '4' },
  ms_comp_side_threshold_db: { id: 's-mscomp-side-thresh',  type: 'number', default: '-20' },
  ms_comp_side_ratio:        { id: 's-mscomp-side-ratio',   type: 'number', default: '4' },
  ms_comp_side_attack_ms:    { id: 's-mscomp-side-attack',  type: 'number', default: '10' },
  ms_comp_side_release_ms:   { id: 's-mscomp-side-release', type: 'number', default: '100' },
  // Multiband
  mb_low_crossover:   { id: 's-mb-lowx',     type: 'number', default: '200' },
  mb_high_crossover:  { id: 's-mb-highx',    type: 'number', default: '2000' },
  mb_low_threshold_db:{ id: 's-mb-low-th',   type: 'number', default: '-20' },
  mb_low_ratio:       { id: 's-mb-low-ratio',type: 'number', default: '4' },
  mb_low_attack_ms:   { id: 's-mb-low-att',  type: 'number', default: '10' },
  mb_low_release_ms:  { id: 's-mb-low-rel',  type: 'number', default: '100' },
  mb_low_makeup_db:   { id: 's-mb-low-mu',   type: 'number', default: '0' },
  mb_mid_threshold_db:{ id: 's-mb-mid-th',   type: 'number', default: '-20' },
  mb_mid_ratio:       { id: 's-mb-mid-ratio',type: 'number', default: '4' },
  mb_mid_attack_ms:   { id: 's-mb-mid-att',  type: 'number', default: '10' },
  mb_mid_release_ms:  { id: 's-mb-mid-rel',  type: 'number', default: '100' },
  mb_mid_makeup_db:   { id: 's-mb-mid-mu',   type: 'number', default: '0' },
  mb_high_threshold_db:{ id: 's-mb-high-th', type: 'number', default: '-20' },
  mb_high_ratio:      { id: 's-mb-high-ratio',type: 'number',default: '4' },
  mb_high_attack_ms:  { id: 's-mb-high-att', type: 'number', default: '10' },
  mb_high_release_ms: { id: 's-mb-high-rel', type: 'number', default: '100' },
  mb_high_makeup_db:  { id: 's-mb-high-mu',  type: 'number', default: '0' },
};

export function collectParams(overrides = null) {
  const params = {};
  for (const [key, def] of Object.entries(PARAM_DEFS)) {
    if (overrides && key in overrides) {
      params[key] = overrides[key];
      continue;
    }
    const el = document.getElementById(def.id);
    if (!el) {
      params[key] = def.default;
      continue;
    }
    if (def.type === 'check') {
      params[key] = el.checked;
    } else if (def.type === 'bool-value') {
      params[key] = el.value === 'true' || el.value === '1';
    } else {
      params[key] = el.value;
    }
  }
  // BUGFIX: limiter_ceiling is stored as dB in UI but backend expects LINEAR (0.5–1.0)
  if (params.limiter_ceiling != null) {
    params.limiter_ceiling = Math.pow(10, Number(params.limiter_ceiling) / 20);
  }
  // Mismo caso que limiter_ceiling: el slider de clipper_ceiling también
  // muestra dB (-6..0) pero el backend espera lineal (0.1–1.0, default 0.98).
  // Sin esto, un valor típico como -0.3 se mandaba tal cual — el backend
  // clampea cualquier negativo a 0.1 (np.clip(ceiling, 0.1, 1.0)), y con
  // ceiling=0.1 el clipper satura brutalmente TODA la señal (soft-clip con
  // tanh(x/0.1)). El clipper además está activo por defecto
  // (#s-clip-bypass vale "0" = no bypasseado), así que esto rompía el
  // preview en cada render, no solo cuando alguien tocaba el módulo.
  if (params.clipper_ceiling != null) {
    params.clipper_ceiling = Math.pow(10, Number(params.clipper_ceiling) / 20);
  }
  return params;
}

export function buildQueryString(params) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') qs.append(k, String(v));
  }
  return qs.toString();
}

export function applyOverridesToUI(overrides) {
  for (const [key, value] of Object.entries(overrides)) {
    const def = PARAM_DEFS[key];
    if (!def) continue;
    const el = document.getElementById(def.id);
    if (!el) continue;
    if (def.type === 'check') {
      el.checked = !!value;
    } else if (def.type === 'bool-value') {
      el.value = value ? 'true' : 'false';
    } else {
      el.value = value;
    }
  }
}
