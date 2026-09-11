"""
Generador de presets completos para mastering.
Expande los presets existentes con todos los parámetros de la cadena.
"""

def create_full_preset(base: dict, **overrides) -> dict:
    """
    Crea un preset completo partiendo de un base y mezclándolo con overrides.
    Asegura que todos los parámetros estén presentes con valores por defecto.
    """
    defaults = {
        # Básicos
        "label": "Unnamed",
        "input_gain_db": 0.0,
        "target_peak": 0.95,
        "use_lufs_normalize": False,
        "target_lufs": -14.0,
        
        # Compresión de banda ancha
        "comp_threshold_db": -18.0,
        "comp_ratio": 4.0,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 100.0,
        "comp_makeup_db": 0.0,
        "comp_pdr": True,
        "comp_pdr_hold_ms": 500.0,
        "comp_stereo_link": True,
        
        # Oversample
        "oversample_mode": "quality",
        
        # Compresión paralela
        "parallel_bypass": True,
        "parallel_threshold_db": -12.0,
        "parallel_ratio": 4.0,
        "parallel_attack_ms": 10.0,
        "parallel_release_ms": 100.0,
        "parallel_mix": 0.0,
        
        # EQ + Shelving
        "hp_cutoff": 30.0,
        "lp_bypass": True,
        "lp_cutoff": 18000.0,
        "high_shelf_gain_db": 0.0,
        "high_shelf_freq_hz": 8000.0,
        "low_shelf_gain_db": 0.0,
        "low_shelf_freq_hz": 100.0,
        
        # Multibanda Stereo Width
        "mb_stereo_bypass": True,
        "mb_stereo_low_width": 0.9,
        "mb_stereo_mid_width": 1.2,
        "mb_stereo_high_width": 1.5,
        "mb_stereo_low_crossover": 150.0,
        "mb_stereo_high_crossover": 4000.0,
        
        # EQs paramétricos
        "eq1_freq": 100.0, "eq1_gain": 0.0, "eq1_q": 1.0,
        "eq2_freq": 500.0, "eq2_gain": 0.0, "eq2_q": 1.0,
        "eq3_freq": 2000.0, "eq3_gain": 0.0, "eq3_q": 1.0,
        "eq4_freq": 8000.0, "eq4_gain": 0.0, "eq4_q": 1.0,
        "eq5_freq": 200.0, "eq5_gain": 0.0, "eq5_q": 1.0,
        "eq6_freq": 1000.0, "eq6_gain": 0.0, "eq6_q": 1.0,
        
        # Transiente
        "transient_attack": 0.0,
        "transient_sustain": 0.0,
        
        # Saturación
        "saturation_drive": 0.0,
        "saturation_mode": "tape",
        "saturation_mix": 1.0,
        
        # M/S
        "mid_gain_db": 0.0,
        "side_gain_db": 0.0,
        "stereo_width_amount": 1.0,
        
        # Stereo Enhancer
        "use_stereo_enhancer": False,
        "enhancer_bass_mono_freq": 120.0,
        "haas_delay_ms": 0.0,
        
        # Reverb
        "reverb_size": 0.3,
        "reverb_wet": 0.0,
        
        # Glue Compressor
        "glue_bypass": True,
        "glue_threshold_db": -4.0,
        "glue_ratio": 2.0,
        "glue_attack_ms": 30.0,
        "glue_release_ms": 120.0,
        "glue_makeup_db": 0.0,
        "glue_pdr": True,
        "glue_pdr_hold_ms": 500.0,
        
        # Limiter
        "limiter_ceiling": 0.95,
        "limiter_release_ms": 50.0,
        
        # EQ Mode
        "eq_mode": "iir",
        "linear_phase_taps": 2049,
        
        # Low-End Mono
        "low_end_mono_freq": 120.0,
        "low_end_mono_amount": 0.0,
        
        # Dynamic EQ
        "dyneq_bypass": True,
        "dyneq_freq": 3000.0,
        "dyneq_q": 2.5,
        "dyneq_threshold_db": -18.0,
        "dyneq_ratio": 3.0,
        "dyneq_attack_ms": 3.0,
        "dyneq_release_ms": 80.0,
        "dyneq_max_reduction_db": 12.0,
        
        # M/S EQ
        "ms_eq_bypass": True,
        "ms_mid_freq": 250.0,
        "ms_mid_gain": 0.0,
        "ms_mid_q": 1.0,
        "ms_side_freq": 8000.0,
        "ms_side_gain": 0.0,
        "ms_side_q": 1.0,
        
        # M/S Compresor
        "ms_comp_bypass": True,
        "ms_comp_mid_threshold_db": -18.0,
        "ms_comp_mid_ratio": 2.0,
        "ms_comp_mid_attack_ms": 15.0,
        "ms_comp_mid_release_ms": 120.0,
        "ms_comp_mid_makeup_db": 0.0,
        "ms_comp_side_threshold_db": -18.0,
        "ms_comp_side_ratio": 2.0,
        "ms_comp_side_attack_ms": 15.0,
        "ms_comp_side_release_ms": 120.0,
        "ms_comp_side_makeup_db": 0.0,
        "ms_comp_pdr": True,
        "ms_comp_pdr_hold_ms": 500.0,
        
        # Resonancia
        "reso_bypass": True,
        "reso_freq": 1200.0,
        "reso_q": 3.0,
        "reso_threshold_db": -18.0,
        "reso_ratio": 3.0,
        "reso_attack_ms": 5.0,
        "reso_release_ms": 100.0,
        "reso_max_reduction_db": 8.0,
        
        # Clipper
        "clipper_bypass": True,
        "clipper_mode": "soft",
        "clipper_ceiling": 0.98,
        "clipper_drive_db": 0.0,
        
        # Noise Reduction
        "nr_bypass": True,
        "nr_strength": 0.5,
        "nr_noise_sample_sec": 0.5,
        
        # Tonal Balance
        "tonal_balance_bypass": True,
        "tonal_balance_amount": 1.0,
        "tonal_balance_max_boost_db": 3.5,
        "tonal_balance_max_cut_db": -4.5,
        "tonal_balance_max_bands": 6,
        
        # Multibanda
        "mb_bypass": False,
        "mb_low_crossover": 250.0,
        "mb_high_crossover": 4000.0,
        "mb_low_threshold_db": -18.0,
        "mb_low_ratio": 2.0,
        "mb_low_attack_ms": 20.0,
        "mb_low_release_ms": 150.0,
        "mb_low_makeup_db": 0.0,
        "mb_mid_threshold_db": -18.0,
        "mb_mid_ratio": 2.0,
        "mb_mid_attack_ms": 20.0,
        "mb_mid_release_ms": 150.0,
        "mb_mid_makeup_db": 0.0,
        "mb_high_threshold_db": -18.0,
        "mb_high_ratio": 2.0,
        "mb_high_attack_ms": 20.0,
        "mb_high_release_ms": 150.0,
        "mb_high_makeup_db": 0.0,
        "mb_pdr": True,
        "mb_pdr_hold_ms": 500.0,
        
        # Output
        "output_format": "wav",
        "output_bit_depth": 24,
        "dither_mode": "f_weighted",
        "platform_target": None,
    }
    
    # Mezclar base
    result = dict(defaults)
    result.update(base)
    result.update(overrides)
    
    return result


# Presets específicos por género (overrides solamente)
ROCK_OVERRIDE = {
    "label": "Rock",
    "target_lufs": -9.5,
    "high_shelf_gain_db": 2.0,
    "high_shelf_freq_hz": 8000.0,
    "eq1_freq": 100.0, "eq1_gain": 1.2, "eq1_q": 1.0,
    "eq2_freq": 400.0, "eq2_gain": -1.2, "eq2_q": 1.1,
    "eq3_freq": 3000.0, "eq3_gain": 1.6, "eq3_q": 1.0,
    "eq4_freq": 9000.0, "eq4_gain": 1.2, "eq4_q": 0.9,
    "comp_threshold_db": -4.7, "comp_ratio": 2.2, "comp_attack_ms": 12.0, "comp_release_ms": 130.0, "comp_makeup_db": 1.2,
    "transient_attack": 0.1, "transient_sustain": 0.05,
    "mb_bypass": False,
    "mb_low_crossover": 150.0, "mb_high_crossover": 4000.0,
    "mb_low_threshold_db": -4.4, "mb_low_ratio": 1.9, "mb_low_attack_ms": 25.0, "mb_low_release_ms": 150.0, "mb_low_makeup_db": 0.4,
    "mb_mid_threshold_db": -3.9, "mb_mid_ratio": 1.8, "mb_mid_attack_ms": 15.0, "mb_mid_release_ms": 120.0, "mb_mid_makeup_db": 0.4,
    "mb_high_threshold_db": -3.3, "mb_high_ratio": 1.6, "mb_high_attack_ms": 8.0, "mb_high_release_ms": 90.0, "mb_high_makeup_db": 0.3,
    "saturation_drive": 0.12, "saturation_mode": "tape", "saturation_mix": 0.25,
    "reverb_size": 0.22, "reverb_wet": 0.03,
    "limiter_ceiling": 0.912, "limiter_release_ms": 70.0,
}

METAL_OVERRIDE = {
    "label": "Metal",
    "target_lufs": -8.5,
    "high_shelf_gain_db": 2.4,
    "high_shelf_freq_hz": 7000.0,
    "eq1_freq": 90.0, "eq1_gain": 1.6, "eq1_q": 1.0,
    "eq2_freq": 300.0, "eq2_gain": -2.4, "eq2_q": 1.3,
    "eq3_freq": 2500.0, "eq3_gain": 2.4, "eq3_q": 1.1,
    "eq4_freq": 8000.0, "eq4_gain": 1.6, "eq4_q": 0.8,
    "comp_threshold_db": -5.7, "comp_ratio": 2.6, "comp_attack_ms": 9.0, "comp_release_ms": 110.0, "comp_makeup_db": 1.6,
    "transient_attack": 0.16, "transient_sustain": 0.0,
    "mb_bypass": False,
    "mb_low_crossover": 180.0, "mb_high_crossover": 3500.0,
    "mb_low_threshold_db": -5.2, "mb_low_ratio": 2.2, "mb_low_attack_ms": 20.0, "mb_low_release_ms": 140.0, "mb_low_makeup_db": 0.4,
    "mb_mid_threshold_db": -4.7, "mb_mid_ratio": 2.0, "mb_mid_attack_ms": 12.0, "mb_mid_release_ms": 100.0, "mb_mid_makeup_db": 0.6,
    "mb_high_threshold_db": -4.0, "mb_high_ratio": 1.8, "mb_high_attack_ms": 6.0, "mb_high_release_ms": 70.0, "mb_high_makeup_db": 0.4,
    "saturation_drive": 0.2, "saturation_mode": "tube", "saturation_mix": 0.28,
    "stereo_width_amount": 1.0,
    "reverb_size": 0.18, "reverb_wet": 0.02,
    "limiter_ceiling": 0.912, "limiter_release_ms": 55.0,
}

TRAP_OVERRIDE = {
    "label": "Trap",
    "target_lufs": -7.5,
    "high_shelf_gain_db": 2.0,
    "high_shelf_freq_hz": 10000.0,
    "eq1_freq": 60.0, "eq1_gain": 2.4, "eq1_q": 0.9,
    "eq2_freq": 250.0, "eq2_gain": -1.6, "eq2_q": 1.2,
    "eq3_freq": 3500.0, "eq3_gain": 2.0, "eq3_q": 1.0,
    "eq4_freq": 10000.0, "eq4_gain": 2.0, "eq4_q": 0.8,
    "comp_threshold_db": -6.7, "comp_ratio": 2.8, "comp_attack_ms": 6.0, "comp_release_ms": 100.0, "comp_makeup_db": 1.6,
    "transient_attack": 0.2, "transient_sustain": -0.08,
    "mb_bypass": False,
    "mb_low_crossover": 100.0, "mb_high_crossover": 4500.0,
    "mb_low_threshold_db": -6.4, "mb_low_ratio": 2.6, "mb_low_attack_ms": 15.0, "mb_low_release_ms": 170.0, "mb_low_makeup_db": 0.8,
    "mb_mid_threshold_db": -4.2, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 10.0, "mb_mid_release_ms": 100.0, "mb_mid_makeup_db": 0.4,
    "mb_high_threshold_db": -3.5, "mb_high_ratio": 1.7, "mb_high_attack_ms": 5.0, "mb_high_release_ms": 60.0, "mb_high_makeup_db": 0.3,
    "saturation_drive": 0.1, "saturation_mode": "tape", "saturation_mix": 0.18,
    "stereo_width_amount": 1.08,
    "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 100.0, "haas_delay_ms": 3.0,
    "reverb_size": 0.18, "reverb_wet": 0.0,
    "limiter_ceiling": 0.912, "limiter_release_ms": 45.0,
}

RAP_OVERRIDE = {
    "label": "Rap / Hip-Hop",
    "target_lufs": -8.5,
    "high_shelf_gain_db": 1.4,
    "high_shelf_freq_hz": 8000.0,
    "eq1_freq": 80.0, "eq1_gain": 2.0, "eq1_q": 0.9,
    "eq2_freq": 350.0, "eq2_gain": -1.2, "eq2_q": 1.1,
    "eq3_freq": 2000.0, "eq3_gain": 1.6, "eq3_q": 1.2,
    "eq4_freq": 7000.0, "eq4_gain": 1.2, "eq4_q": 0.9,
    "comp_threshold_db": -5.0, "comp_ratio": 2.2, "comp_attack_ms": 11.0, "comp_release_ms": 115.0, "comp_makeup_db": 1.2,
    "transient_attack": 0.13, "transient_sustain": 0.0,
    "mb_bypass": False,
    "mb_low_crossover": 120.0, "mb_high_crossover": 4000.0,
    "mb_low_threshold_db": -5.4, "mb_low_ratio": 2.2, "mb_low_attack_ms": 18.0, "mb_low_release_ms": 160.0, "mb_low_makeup_db": 0.5,
    "mb_mid_threshold_db": -4.2, "mb_mid_ratio": 1.8, "mb_mid_attack_ms": 12.0, "mb_mid_release_ms": 110.0, "mb_mid_makeup_db": 0.4,
    "mb_high_threshold_db": -3.5, "mb_high_ratio": 1.6, "mb_high_attack_ms": 6.0, "mb_high_release_ms": 80.0, "mb_high_makeup_db": 0.3,
    "saturation_drive": 0.1, "saturation_mode": "tape", "saturation_mix": 0.2,
    "reverb_size": 0.18, "reverb_wet": 0.03,
    "limiter_ceiling": 0.912, "limiter_release_ms": 65.0,
}

REGGAETON_OVERRIDE = {
    "label": "Reggaeton",
    "target_lufs": -8.0,
    "high_shelf_gain_db": 2.0,
    "high_shelf_freq_hz": 9000.0,
    "eq1_freq": 70.0, "eq1_gain": 2.4, "eq1_q": 0.9,
    "eq2_freq": 300.0, "eq2_gain": -1.6, "eq2_q": 1.2,
    "eq3_freq": 3000.0, "eq3_gain": 2.0, "eq3_q": 1.0,
    "eq4_freq": 9000.0, "eq4_gain": 1.6, "eq4_q": 0.8,
    "comp_threshold_db": -6.0, "comp_ratio": 2.5, "comp_attack_ms": 9.0, "comp_release_ms": 105.0, "comp_makeup_db": 1.4,
    "transient_attack": 0.17, "transient_sustain": -0.04,
    "mb_bypass": False,
    "mb_low_crossover": 110.0, "mb_high_crossover": 4200.0,
    "mb_low_threshold_db": -5.7, "mb_low_ratio": 2.4, "mb_low_attack_ms": 16.0, "mb_low_release_ms": 170.0, "mb_low_makeup_db": 0.6,
    "mb_mid_threshold_db": -4.4, "mb_mid_ratio": 1.9, "mb_mid_attack_ms": 10.0, "mb_mid_release_ms": 110.0, "mb_mid_makeup_db": 0.4,
    "mb_high_threshold_db": -3.6, "mb_high_ratio": 1.7, "mb_high_attack_ms": 6.0, "mb_high_release_ms": 75.0, "mb_high_makeup_db": 0.3,
    "saturation_drive": 0.1, "saturation_mode": "tape", "saturation_mix": 0.2,
    "stereo_width_amount": 1.12,
    "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 110.0, "haas_delay_ms": 3.0,
    "reverb_size": 0.15, "reverb_wet": 0.02,
    "limiter_ceiling": 0.912, "limiter_release_ms": 50.0,
}

POP_OVERRIDE = {
    "label": "Pop",
    "target_lufs": -10.0,
    "high_shelf_gain_db": 2.0,
    "high_shelf_freq_hz": 10000.0,
    "eq1_freq": 100.0, "eq1_gain": 0.8, "eq1_q": 1.0,
    "eq2_freq": 400.0, "eq2_gain": -0.8, "eq2_q": 1.1,
    "eq3_freq": 3000.0, "eq3_gain": 1.6, "eq3_q": 1.0,
    "eq4_freq": 10000.0, "eq4_gain": 2.0, "eq4_q": 0.8,
    "comp_threshold_db": -4.4, "comp_ratio": 2.0, "comp_attack_ms": 13.0, "comp_release_ms": 135.0, "comp_makeup_db": 1.0,
    "transient_attack": 0.08, "transient_sustain": 0.0,
    "mb_bypass": False,
    "mb_low_crossover": 200.0, "mb_high_crossover": 4000.0,
    "mb_low_threshold_db": -3.9, "mb_low_ratio": 1.8, "mb_low_attack_ms": 22.0, "mb_low_release_ms": 150.0, "mb_low_makeup_db": 0.3,
    "mb_mid_threshold_db": -3.3, "mb_mid_ratio": 1.6, "mb_mid_attack_ms": 14.0, "mb_mid_release_ms": 120.0, "mb_mid_makeup_db": 0.3,
    "mb_high_threshold_db": -3.1, "mb_high_ratio": 1.5, "mb_high_attack_ms": 8.0, "mb_high_release_ms": 90.0, "mb_high_makeup_db": 0.2,
    "saturation_drive": 0.07, "saturation_mode": "tape", "saturation_mix": 0.16,
    "stereo_width_amount": 1.08,
    "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 120.0, "haas_delay_ms": 3.0,
    "reverb_size": 0.28, "reverb_wet": 0.05,
    "limiter_ceiling": 0.912, "limiter_release_ms": 75.0,
}

CD_OVERRIDE = {
    "label": "CD / Audiófilo",
    "target_lufs": -16.0,
    "high_shelf_gain_db": 0.6,
    "high_shelf_freq_hz": 8000.0,
    "eq1_freq": 100.0, "eq1_gain": 0.0, "eq1_q": 1.0,
    "eq2_freq": 500.0, "eq2_gain": 0.0, "eq2_q": 1.0,
    "eq3_freq": 2000.0, "eq3_gain": 0.3, "eq3_q": 1.0,
    "eq4_freq": 9000.0, "eq4_gain": 0.3, "eq4_q": 0.9,
    "comp_threshold_db": -1.7, "comp_ratio": 1.4, "comp_attack_ms": 22.0, "comp_release_ms": 190.0, "comp_makeup_db": 0.0,
    "transient_attack": 0.0, "transient_sustain": 0.0,
    "mb_bypass": True,
    "saturation_drive": 0.0, "saturation_mix": 0.0,
    "reverb_wet": 0.02,
    "limiter_ceiling": 0.891, "limiter_release_ms": 110.0,
}

EDM_OVERRIDE = {
    "label": "EDM / House",
    "target_lufs": -7.0,
    "high_shelf_gain_db": 2.2,
    "high_shelf_freq_hz": 9000.0,
    "eq1_freq": 75.0, "eq1_gain": 2.0, "eq1_q": 0.9,
    "eq2_freq": 280.0, "eq2_gain": -1.4, "eq2_q": 1.2,
    "eq3_freq": 3200.0, "eq3_gain": 1.8, "eq3_q": 1.0,
    "eq4_freq": 9500.0, "eq4_gain": 1.8, "eq4_q": 0.8,
    "comp_threshold_db": -6.2, "comp_ratio": 2.4, "comp_attack_ms": 7.0, "comp_release_ms": 105.0, "comp_makeup_db": 1.4,
    "transient_attack": 0.15, "transient_sustain": 0.0,
    "mb_bypass": False,
    "mb_low_crossover": 100.0, "mb_high_crossover": 4500.0,
    "mb_low_threshold_db": -5.9, "mb_low_ratio": 2.3, "mb_low_attack_ms": 16.0, "mb_low_release_ms": 150.0, "mb_low_makeup_db": 0.5,
    "mb_mid_threshold_db": -4.5, "mb_mid_ratio": 2.0, "mb_mid_attack_ms": 10.0, "mb_mid_release_ms": 100.0, "mb_mid_makeup_db": 0.4,
    "mb_high_threshold_db": -3.8, "mb_high_ratio": 1.7, "mb_high_attack_ms": 5.0, "mb_high_release_ms": 70.0, "mb_high_makeup_db": 0.3,
    "saturation_drive": 0.14, "saturation_mode": "tube", "saturation_mix": 0.26,
    "side_gain_db": 0.4, "stereo_width_amount": 1.1,
    "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 105.0, "haas_delay_ms": 2.0,
    "reverb_size": 0.2, "reverb_wet": 0.04,
    "limiter_ceiling": 0.912, "limiter_release_ms": 60.0,
}

AMBIENT_OVERRIDE = {
    "label": "Ambient / Chill",
    "target_lufs": -15.0,
    "high_shelf_gain_db": 0.8,
    "high_shelf_freq_hz": 10000.0,
    "eq1_freq": 90.0, "eq1_gain": 0.4, "eq1_q": 1.0,
    "eq2_freq": 400.0, "eq2_gain": -0.6, "eq2_q": 1.0,
    "eq3_freq": 2500.0, "eq3_gain": 0.6, "eq3_q": 1.0,
    "eq4_freq": 10000.0, "eq4_gain": 1.0, "eq4_q": 0.8,
    "comp_threshold_db": -2.4, "comp_ratio": 1.4, "comp_attack_ms": 25.0, "comp_release_ms": 210.0, "comp_makeup_db": 0.3,
    "transient_attack": 0.0, "transient_sustain": 0.1,
    "mb_bypass": False,
    "mb_low_crossover": 170.0, "mb_high_crossover": 4200.0,
    "mb_low_threshold_db": -1.9, "mb_low_ratio": 1.3, "mb_low_attack_ms": 32.0, "mb_low_release_ms": 220.0, "mb_low_makeup_db": 0.0,
    "mb_mid_threshold_db": -1.7, "mb_mid_ratio": 1.3, "mb_mid_attack_ms": 24.0, "mb_mid_release_ms": 190.0, "mb_mid_makeup_db": 0.0,
    "mb_high_threshold_db": -1.4, "mb_high_ratio": 1.25, "mb_high_attack_ms": 14.0, "mb_high_release_ms": 140.0, "mb_high_makeup_db": 0.0,
    "saturation_drive": 0.05, "saturation_mix": 0.12,
    "side_gain_db": 0.6, "stereo_width_amount": 1.15,
    "use_stereo_enhancer": True, "enhancer_bass_mono_freq": 100.0, "haas_delay_ms": 6.0,
    "reverb_size": 0.55, "reverb_wet": 0.14,
    "limiter_ceiling": 0.867, "limiter_release_ms": 180.0,
}

PODCAST_OVERRIDE = {
    "label": "Podcast / Voz",
    "use_lufs_normalize": True,
    "target_lufs": -16.0,
    "hp_cutoff": 80.0,
    "high_shelf_gain_db": 1.0,
    "eq1_freq": 120.0, "eq1_gain": -1.0, "eq1_q": 1.0,
    "eq2_freq": 350.0, "eq2_gain": -1.5, "eq2_q": 1.2,
    "eq3_freq": 3200.0, "eq3_gain": 2.0, "eq3_q": 1.1,
    "eq4_freq": 8000.0, "eq4_gain": 1.0, "eq4_q": 0.9,
    "comp_threshold_db": -5.2, "comp_ratio": 2.6, "comp_attack_ms": 8.0, "comp_release_ms": 120.0, "comp_makeup_db": 1.5,
    "transient_attack": 0.0, "transient_sustain": 0.0,
    "mb_bypass": True,
    "saturation_drive": 0.0, "saturation_mix": 0.0,
    "reverb_wet": 0.0,
    "limiter_ceiling": 0.891, "limiter_release_ms": 90.0,
}

# Presets profesionales LGMDM — catálogo compacto de 6 perfiles.
# Todos parten del mismo preset base completo y aplican overrides de género.
URBAN_OVERRIDE = dict(RAP_OVERRIDE, label="Urban", target_lufs=-9.0,
    high_shelf_gain_db=1.6, high_shelf_freq_hz=8500.0, stereo_width_amount=1.05)
CUMBIA_OVERRIDE = dict(REGGAETON_OVERRIDE, label="Cumbia", target_lufs=-9.0,
    eq2_freq=420.0, eq2_gain=-1.0, eq3_freq=2200.0, eq3_gain=1.4,
    stereo_width_amount=1.08, reverb_wet=0.03)
CUARTETO_OVERRIDE = dict(POP_OVERRIDE, label="Cuarteto", target_lufs=-9.5,
    eq1_freq=110.0, eq1_gain=1.2, eq2_freq=500.0, eq2_gain=-1.0,
    eq3_freq=2600.0, eq3_gain=1.8, stereo_width_amount=1.06, saturation_drive=0.09)
LATINO_OVERRIDE = dict(REGGAETON_OVERRIDE, label="Latino", target_lufs=-9.0,
    high_shelf_gain_db=1.8, eq2_freq=360.0, eq2_gain=-1.2,
    eq3_freq=2800.0, eq3_gain=1.8, stereo_width_amount=1.10, reverb_wet=0.025)

# Build final presets
MASTERING_PRESETS_FULL = {
    "rock": create_full_preset({}, **ROCK_OVERRIDE),
    "pop": create_full_preset({}, **POP_OVERRIDE),
    "urban": create_full_preset({}, **URBAN_OVERRIDE),
    "cumbia": create_full_preset({}, **CUMBIA_OVERRIDE),
    "cuarteto": create_full_preset({}, **CUARTETO_OVERRIDE),
    "latino": create_full_preset({}, **LATINO_OVERRIDE),
}
