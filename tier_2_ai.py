# Copyright (c) 2026 Vrishank Yadav
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import os
import time

# ==========================================
# HYPERPARAMETER CONTROLS & CONFIGURATION
# ==========================================
X_DATA_PATH = "X_spectra_ready.npy"
TIER1_RESULTS_PATH = "tier1_results.npy"

WL_START = 340.0
WL_END = 1050.0

# --- SPECTRUM LINE CORE CONFIGURATION ---
LAMBDA_CA_K       = 393.3
LAMBDA_CA_H       = 396.8
LAMBDA_H_EPSILON  = 397.007  # Hydrogen Epsilon
LAMBDA_H_DELTA    = 410.2    # Hydrogen Delta
LAMBDA_H_GAMMA    = 434.0    # Hydrogen Gamma
LAMBDA_H_BETA     = 486.1    # Hydrogen Beta
LAMBDA_H_ALPHA    = 656.3    # Hydrogen Alpha
LAMBDA_HE_I_BLUE  = 447.1    # Neutral Helium (He I)
LAMBDA_HE_II_DEEP = 468.6    # Ionized Helium (He II)

# --- FINE TUNING DIALS ---
TOTAL_T2_PASSES = 200       # Upgraded optimization passes
LINE_WINDOW_NM = 10.0       # Physical width (nm) around atomic targets
LEARNING_RATE = 0.50        # Blending memory weight factor

# --- SUPER SENSITIVITY BIAS ---
SENSITIVITY_BIAS = 0.27    

print("==================================================================")
print("     LAUNCHING SKEPTIC TIER 2 RECURSIVE ATOMIC FEATURE ENGINE     ")
print("==================================================================")

# ==========================================
# 1. LOAD DATA & SYNCHRONIZE DEPENDENCIES
# ==========================================
if not os.path.exists(X_DATA_PATH):
    X_DATA_PATH = os.path.join("gaia_data_sort", "X_spectra_ready.npy")

if not os.path.exists(TIER1_RESULTS_PATH):
    raise FileNotFoundError("Missing 'tier1_results.npy'. Run Tier 1 Map Engine first!")

X = np.load(X_DATA_PATH)
X_flattened = X.reshape(X.shape[0], -1)
num_stars, num_features = X_flattened.shape
wavelengths = np.linspace(WL_START, WL_END, num_features)

# Load the upfront assignments from Tier 1
tier1_assignments = np.load(TIER1_RESULTS_PATH)

# ==========================================
# 2. LINEAR DETRENDING WINDOW EXTRACTOR
# ==========================================
def extract_local_window(target_nm):
    """Isolates, detrends, and flattens local wavelength windows to isolate pure line depth."""
    idx_center = np.abs(wavelengths - target_nm).argmin()
    idx_delta = int((LINE_WINDOW_NM / (WL_END - WL_START)) * num_features)
    start_idx = max(0, idx_center - idx_delta)
    end_idx = min(num_features, idx_center + idx_delta)
    
    raw_windows = X_flattened[:, start_idx:end_idx]
    num_local_points = raw_windows.shape[1]
    x_indices = np.arange(num_local_points)
    
    detrended_windows = np.zeros_like(raw_windows)
    for idx in range(num_stars):
        y_signals = raw_windows[idx]
        slope_fit = np.polyfit(x_indices, y_signals, 1)
        background_trend = np.polyval(slope_fit, x_indices)
        detrended_windows[idx] = (y_signals - background_trend) / (np.mean(y_signals) + 1e-8)
        
    return detrended_windows

print("\nExecuting physical detrending sweep across target line profiles...")
pockets = {
    "ca_k":       extract_local_window(LAMBDA_CA_K),
    "ca_h":       extract_local_window(LAMBDA_CA_H),
    "h_epsilon":  extract_local_window(LAMBDA_H_EPSILON),
    "h_delta":    extract_local_window(LAMBDA_H_DELTA),
    "h_gamma":    extract_local_window(LAMBDA_H_GAMMA),
    "h_beta":     extract_local_window(LAMBDA_H_BETA),
    "h_alpha":    extract_local_window(LAMBDA_H_ALPHA),
    "he_i":       extract_local_window(LAMBDA_HE_I_BLUE),
    "he_ii":      extract_local_window(LAMBDA_HE_II_DEEP)
}

# ==========================================
# 3. RECURSIVE MONITORING ENGINE
# ==========================================
def optimize_feature_archetypes(feature_matrix, feature_label):
    print(f"\n[SCANNING METRIC: {feature_label.upper()}]")
    print("-" * 50)
    
    mid_point = feature_matrix.shape[1] // 2
    core_depths = feature_matrix[:, mid_point]
    sorted_depths = np.argsort(core_depths)
    
    # Seed initial cluster memories
    arch_flat = np.mean(feature_matrix[sorted_depths[-int(num_stars*0.25):]], axis=0)
    arch_active = np.mean(feature_matrix[sorted_depths[:int(num_stars*0.25)]], axis=0)
    current_archetypes = np.stack([arch_flat, arch_active])
    
    last_assignments = np.zeros(num_stars)
    
    # 200-Pass Optimization Cycle
    for pass_idx in range(1, TOTAL_T2_PASSES + 1):
        dist_0 = np.linalg.norm(feature_matrix - current_archetypes[0], axis=1)
        dist_1 = np.linalg.norm(feature_matrix - current_archetypes[1], axis=1)
        
        # Apply super-sensitivity bias modifier scaling
        effective_dist_1 = dist_1 * (1.0 - SENSITIVITY_BIAS)
        assignments = (effective_dist_1 < dist_0).astype(int)
        
        num_present = np.sum(assignments == 1)
        pct_present = (num_present / num_stars) * 100
        swapped = np.sum(assignments != last_assignments) if pass_idx > 1 else num_stars
        
        if pass_idx == 1 or pass_idx % 25 == 0 or pass_idx == TOTAL_T2_PASSES:
            print(f" Pass {pass_idx:03d}/{TOTAL_T2_PASSES} -> Present: {num_present:5d} stars ({pct_present:5.1f}%) | Flux Drift: {swapped:4d}")
        
        last_assignments = np.copy(assignments)
        
        for arch_idx in [0, 1]:
            mask = (assignments == arch_idx)
            if np.sum(mask) > 10:
                observed_mean = np.mean(feature_matrix[mask], axis=0)
                current_archetypes[arch_idx] = (current_archetypes[arch_idx] * (1.0 - LEARNING_RATE)) + (observed_mean * LEARNING_RATE)
                
    # Generate finalized parameters
    dist_0 = np.linalg.norm(feature_matrix - current_archetypes[0], axis=1)
    dist_1 = np.linalg.norm(feature_matrix - current_archetypes[1], axis=1)
    effective_dist_1 = dist_1 * (1.0 - SENSITIVITY_BIAS)
    final_assignments = (effective_dist_1 < dist_0).astype(int)
    
    max_d = np.maximum(dist_0, effective_dist_1) + 1e-8
    margin = np.abs(dist_0 - effective_dist_1) / max_d
    final_confidences = 50.0 + (margin * 49.99)
    
    return final_assignments, final_confidences

# Compute full sequence profile states
h_eps_pres, h_eps_conf = optimize_feature_archetypes(pockets["h_epsilon"], "Hydrogen-Epsilon")
h_del_pres, h_del_conf = optimize_feature_archetypes(pockets["h_delta"],   "Hydrogen-Delta")
h_gam_pres, h_gam_conf = optimize_feature_archetypes(pockets["h_gamma"],   "Hydrogen-Gamma")
h_bet_pres, h_bet_conf = optimize_feature_archetypes(pockets["h_beta"],    "Hydrogen-Beta")
h_alp_pres, h_alp_conf = optimize_feature_archetypes(pockets["h_alpha"],   "Hydrogen-Alpha")
he_i_pres,  he_i_conf  = optimize_feature_archetypes(pockets["he_i"],      "Helium-I")
he_ii_pres, he_ii_conf = optimize_feature_archetypes(pockets["he_ii"],     "Helium-II")
cah_pres,   cah_conf   = optimize_feature_archetypes(pockets["ca_h"],      "Calcium-H")
cak_pres,   cak_conf   = optimize_feature_archetypes(pockets["ca_k"],      "Calcium-K")

# ==========================================
# 4. SKEPTIC SUBCLASS RE-EVALUATION LOOP
# ==========================================
print("\n[RE-EVALUATING SUBCLASSES FROM LOCALIZED FEATURES]")
print("-" * 50)

subclasses = np.zeros(num_stars, dtype=int)
# Expand matrix dimensions to track all 9 key line systems uniformly 
line_metrics = np.zeros((num_stars, 20)) 

# Evaluate atomic tracking arrays across the targets
# We stack total combined local Balmer + Helium metrics to compute precise localized ranking
atomic_strength_score = (h_alp_pres * 1.5) + h_bet_pres + h_gam_pres + h_del_pres + (h_eps_pres * 1.2) + (he_i_pres * 2.0) + (he_ii_pres * 2.5)

for i in range(num_stars):
    # Grab assignment from Tier 1 Results Handshake
    t1_class = tier1_assignments[i]
    
    # Isolate only peers assigned the same group by Tier 1 to re-rank inside the slice
    peer_mask = (tier1_assignments == t1_class)
    peer_scores = atomic_strength_score[peer_mask]
    
    # Calculate relative profile metric depth percentile positioning
    if np.sum(peer_mask) > 1:
        rank_percentile = np.sum(peer_scores < atomic_strength_score[i]) / np.sum(peer_mask)
    else:
        rank_percentile = 0.5
        
    # Scale ranking linearly to create the precise 0-9 Harvard subclass value
    subclasses[i] = int(np.clip(rank_percentile * 10, 0, 9))
    
    # Compile multi-column performance signature array
    line_metrics[i] = [
        t1_class,            subclasses[i],
        h_alp_pres[i],       h_alp_conf[i],
        h_bet_pres[i],       h_bet_conf[i],
        h_gam_pres[i],       h_gam_conf[i],
        h_del_pres[i],       h_del_conf[i],
        h_eps_pres[i],       h_eps_conf[i],
        he_i_pres[i],        he_i_conf[i],
        he_ii_pres[i],       he_ii_conf[i],
        cah_pres[i],         cah_conf[i],
        cak_pres[i],         cak_conf[i]
    ]

# ==========================================
# 5. INTEGRATED COMPILATION EXPORT
# ==========================================
print("\nExporting synchronized data models to storage...")
np.save("tier2_subclasses.npy", subclasses)
np.save("tier2_line_metrics.npy", line_metrics)
print("🎉 Tier 2 optimization complete. Output files published for Tier 3 matching matrix.")
