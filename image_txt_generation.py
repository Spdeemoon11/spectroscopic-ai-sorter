# Copyright (c) 2026 Vrishank Yadav
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION & DIRECTORY MANAGEMENT
# ==========================================
X_DATA_PATH = "X_spectra_ready.npy"
TIER2_DATA_PATH = "tier2_line_metrics.npy"
TIER3_DATA_PATH = "tier3_results.npy"
TIER4_DATA_PATH = "tier4_final_census.csv"

# Zooniverse workspace structural layout
BASE_OUT_DIR = "zooniverse_preview"
IMAGE_DIR = os.path.join(BASE_OUT_DIR, "images")
METADATA_DIR = os.path.join(BASE_OUT_DIR, "metadata")

for folder in [IMAGE_DIR, METADATA_DIR]:
    os.makedirs(folder, exist_ok=True)

# Astrophysical Constants
WL_START, WL_END = 300.0, 1000.0
LAMBDA_CA_K = 393.3
LAMBDA_CA_H = 396.8
LAMBDA_H_BETA = 486.1
LAMBDA_H_ALPHA = 656.3

SPECTRAL_CLASSES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
LUMINOSITY_CLASSES = ['I', 'II', 'III', 'IV', 'V']

print("[SYS] Initializing Streamlined Zooniverse Upload Pipeline...")

# Local file validation checks
paths = [X_DATA_PATH, TIER2_DATA_PATH, TIER3_DATA_PATH, TIER4_DATA_PATH]
for i, path in enumerate(paths):
    if not os.path.exists(path):
        alt_path = os.path.join("gaia_data_sort", path)
        if os.path.exists(alt_path):
            if i == 0: X_DATA_PATH = alt_path
            if i == 1: TIER2_DATA_PATH = alt_path
            if i == 2: TIER3_DATA_PATH = alt_path
            if i == 3: TIER4_DATA_PATH = alt_path
        else:
            raise FileNotFoundError(f"Missing essential pipeline dependency file: {path}")

# ==========================================
# 1. LOAD DATA VECTORS & CALIBRATE METRICS
# ==========================================
print("[SYS] Loading database vectors...")
X = np.load(X_DATA_PATH)
X_flattened = X.reshape(X.shape[0], -1)
num_stars, num_features = X_flattened.shape

t2_metrics = np.load(TIER2_DATA_PATH)
t3_results = np.load(TIER3_DATA_PATH)
t4_df = pd.read_csv(TIER4_DATA_PATH)

wavelengths = np.linspace(WL_START, WL_END, num_features)

# Extract core model predictions
t4_spec_idx = t4_df['spectral_class_idx'].values
t4_lum_idx = t4_df['luminosity_class_idx'].values
t4_certainty = t4_df['fused_certainty_pct'].values

# Calculate operational continuous subclasses
spectral_density = np.sum(t2_metrics[:, [2, 4, 6, 8]], axis=1)
max_density = np.max(spectral_density) if np.max(spectral_density) > 0 else 1.0
t2_subclasses = np.clip(np.floor((spectral_density / max_density) * 9).astype(int), 0, 9)

# Extract confidence bounds across line vectors
h_beta_conf_arr = t2_metrics[:, 7] * 100 if np.max(t2_metrics[:, 7]) <= 1.0 else t2_metrics[:, 7]
h_alpha_conf_arr = t2_metrics[:, 9] * 100 if np.max(t2_metrics[:, 9]) <= 1.0 else t2_metrics[:, 9]

# ==========================================
# 2. BURST CONFIGURATION LOOP
# ==========================================
START_INDEX = 0
BURST_SIZE = 1000  # Control chunk size easily right here
END_INDEX = min(START_INDEX + BURST_SIZE, num_stars)

print(f"[BURST] Processing subject slice window: Indices {START_INDEX} to {END_INDEX - 1}")
manifest_rows = []

for idx in range(START_INDEX, END_INDEX):
    star_id = f"star_{idx:05d}"
    spectrum = X_flattened[idx]
    
    # Resolve physical designations for text profile generations
    spec_class = SPECTRAL_CLASSES[t4_spec_idx[idx]]
    subclass_digit = t2_subclasses[idx]
    lum_class = LUMINOSITY_CLASSES[t4_lum_idx[idx]]
    
    ca_h_present = int(t2_metrics[idx, 2]) == 1
    ca_k_present = int(t2_metrics[idx, 4]) == 1
    h_beta_present = int(t2_metrics[idx, 6]) == 1
    h_alpha_present = int(t2_metrics[idx, 8]) == 1
    
    h_alpha_str = "present" if h_alpha_present else "not present"
    h_beta_str = "present" if h_beta_present else "not present"
    
    # --- CHART GENERATION 1: PLAIN SPECTROGRAPH ---
    plain_img_name = f"{star_id}_spectrograph_plain.png"
    plain_img_path = os.path.join(IMAGE_DIR, plain_img_name)
    
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectrum, color='darkturquoise', linewidth=1.5)
    plt.title(f"Stellar Spectrograph Signal Profile ({star_id})", fontsize=10, fontweight='bold')
    plt.xlabel("Wavelength Scale (nm)")
    plt.ylabel("Normalized Flux")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plain_img_path, dpi=100)
    plt.close()
    
    # --- CHART GENERATION 2: HIGHLIGHTED BALMER SERIES ---
    balmer_img_name = f"{star_id}_spectrograph_balmer.png"
    balmer_img_path = os.path.join(IMAGE_DIR, balmer_img_name)
    
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectrum, color='darkturquoise', linewidth=1.5, label="Raw Profile")
    
    if h_alpha_present:
        plt.axvspan(LAMBDA_H_ALPHA - 5, LAMBDA_H_ALPHA + 5, color='orange', alpha=0.25, label="H-α Line")
        plt.text(LAMBDA_H_ALPHA, np.max(spectrum)*0.92, 'H-α', color='darkorange', ha='center', fontsize=8, fontweight='bold')
        
    if h_beta_present:
        plt.axvspan(LAMBDA_H_BETA - 5, LAMBDA_H_BETA + 5, color='limegreen', alpha=0.25, label="H-β Line")
        plt.text(LAMBDA_H_BETA, np.max(spectrum)*0.92, 'H-β', color='forestgreen', ha='center', fontsize=8, fontweight='bold')
        
    if ca_h_present:
        plt.axvspan(LAMBDA_CA_H - 4, LAMBDA_CA_H + 4, color='darkviolet', alpha=0.25, label="Ca-H Line")
        plt.text(LAMBDA_CA_H, np.max(spectrum)*0.85, 'Ca-H', color='darkviolet', ha='center', fontsize=8, fontweight='bold')

    plt.title(f"Balmer Line Isolation Map ({star_id} - Class: {spec_class}{subclass_digit}{lum_class})", fontsize=10, fontweight='bold')
    plt.xlabel("Wavelength Scale (nm)")
    plt.ylabel("Normalized Flux")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(balmer_img_path, dpi=100)
    plt.close()
    
    # --- OUTPUT ASSET 3: COMPREHENSIVE TEXT METADATA FILE ---
    txt_name = f"{star_id}_metadata.txt"
    txt_path = os.path.join(METADATA_DIR, txt_name)
    
    ledger_content = (
        f"star_id- {idx}\n"
        f"Star class- {spec_class}\n"
        f"Star Subclass- {subclass_digit}\n"
        f"Star Luminousity- {lum_class}\n"
        f"calcium H- {t2_metrics[idx, 2]:.4f}\n"
        f"calcium K- {t2_metrics[idx, 4]:.4f}\n"
        f"hydrogen Alpha- {h_alpha_str} ({h_alpha_conf_arr[idx]:.1f}%)\n"
        f"Hydrogen Beta- {h_beta_str} ({h_beta_conf_arr[idx]:.1f}%)\n"
        f"Hydrogen Elipson- not present (0.0%)\n"
    )
    
    with open(txt_path, "w") as f:
        f.write(ledger_content)
        
    # --- COMPILE MINIMAL ROUTING ENTRY FOR MANIFEST ---
    manifest_rows.append({
        "subject_id": star_id,
        "media1": os.path.join("images", plain_img_name),
        "media2": os.path.join("images", balmer_img_name),
        "meta_file_ref": os.path.join("metadata", txt_name)
    })

# ==========================================
# 3. EXPORT CLEAN MANIFEST SHEETS
# ==========================================
print("\n[EXPORT] Baking clean routing manifest...")
manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(os.path.join(BASE_OUT_DIR, "preview_manifest.csv"), index=False)

# Build volume analysis CSV ledger
summary_rows = [{
    "Total_Pipeline_Stars": num_stars,
    "Current_Burst_Processed": len(manifest_rows),
    "Images_Generated": len(manifest_rows) * 2,
    "Metadata_Ledgers_Generated": len(manifest_rows)
}]
summary_df = pd.DataFrame(summary_rows)
summary_df
