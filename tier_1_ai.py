# Copyright (c) 2026 Vrishank Yadav
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
TOTAL_PASSES = 200  
CLASS_NAMES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

# ==========================================
# 1. LOAD DATA MATRIX
# ==========================================
print("Loading preprocessed matrix...")
x_path = "X_spectra_ready.npy"

if not os.path.exists(x_path):
    x_path = os.path.join("gaia_data_sort", "X_spectra_ready.npy")

if not os.path.exists(x_path):
    raise FileNotFoundError("Could not find 'X_spectra_ready.npy'.")

X = np.load(x_path)
X_flattened = X.reshape(X.shape[0], -1)
num_stars, num_features = X_flattened.shape
print(f"Successfully loaded {num_stars} stars with {num_features} spectral features.")

# ==========================================
# 2. DATA-DRIVEN PERCENTILE ANCHOR EXTRACTION
# ==========================================
print("\nExtracting distribution anchors directly from dataset boundaries...")

# Calculate a single signature score for each star based on its feature density
# This ranks the data from highest energy/density to lowest energy/density
feature_densities = np.sum(X_flattened, axis=1)
sorted_indices = np.argsort(feature_densities)

# Slice the real data into 7 clean progressive percentiles
# This completely prevents a single-bucket collapse because it seeds the memory evenly
percentile_slices = np.linspace(0, num_stars - 1, 8, dtype=int)
ideal_anchors = np.zeros((7, num_features))

for idx in range(7):
    start_idx = percentile_slices[idx]
    end_idx = percentile_slices[idx + 1]
    # The initial anchor memory is the true mathematical average of that dataset slice
    ideal_anchors[idx] = np.mean(X_flattened[sorted_indices[start_idx:end_idx]], axis=0)

# ==========================================
# 3. MULTI-PASS COMPONENT REFINEMENT LOOP
# ==========================================
print(f"\n[STEP 1/3] Launching Percentile-Anchored Sorting over {TOTAL_PASSES} passes...")
current_anchors = np.copy(ideal_anchors)

for pass_idx in range(1, TOTAL_PASSES + 1):
    print(f" -> Processing Pass {pass_idx}/{TOTAL_PASSES}...")
    time.sleep(0.05) # Slipped down processing block latency slightly
    
    # Calculate Euclidean distance to the data-driven anchors
    distances = np.zeros((num_stars, 7))
    for idx in range(7):
        distances[:, idx] = np.linalg.norm(X_flattened - current_anchors[idx], axis=1)
        
    # Assign stars to the best fitting anchor structure
    final_assignments = np.argmin(distances, axis=1)
    
    # Evolve the gut feeling memory by checking what landed in each bucket
    for idx in range(7):
        assigned_stars = X_flattened[final_assignments == idx]
        if len(assigned_stars) > 0:
            observed_mean = np.mean(assigned_stars, axis=0)
            # Blend 20% of current observations into 80% of structural memory
            current_anchors[idx] = (current_anchors[idx] * 0.80) + (observed_mean * 0.20)
        else:
            # Emergency reset if a bucket goes completely empty: re-seed from original slice
            start_idx = percentile_slices[idx]
            end_idx = percentile_slices[idx + 1]
            current_anchors[idx] = np.mean(X_flattened[sorted_indices[start_idx:end_idx]], axis=0)

print("\nRefinement loop locked. Generating final census distribution...")

# ==========================================
# 4. SIMILARITY METRIC COMPILATION
# ==========================================
final_distances = np.zeros((num_stars, 7))
for idx in range(7):
    final_distances[:, idx] = np.linalg.norm(X_flattened - current_anchors[idx], axis=1)

min_distances = np.min(final_distances, axis=1)
max_dist = np.max(min_distances) if np.max(min_distances) > 0 else 1.0
similarities = (1.0 - (min_distances / max_dist)) * 100

counts = []
avg_similarities = []

for idx in range(7):
    mask = (final_assignments == idx)
    current_count = np.sum(mask)
    counts.append(current_count)
    if current_count > 0:
        avg_similarities.append(np.mean(similarities[mask]))
    else:
        avg_similarities.append(0.0)

# ==========================================
# 5. PERSONAL BINARY FILE EXPORT HANDSHAKE
# ==========================================
print("\n[STEP 2/3] Exporting Tier 1 personal published version...")
# Saves the absolute integer matrix array (0 through 6 mapping directly to O-M)
np.save("tier1_results.npy", final_assignments)
print("🎉 Success! Published 'tier1_results.npy' for Tier 2 extraction logic.")

# ==========================================
# 6. HISTOGRAM RENDER STEP
# ==========================================
print("\n[STEP 3/3] Generating final data-calibrated stellar census visualization...")

plt.figure(figsize=(12, 6))
colors = ['#0000FF', '#4B0082', '#8A2BE2', '#40E0D0', '#FFD700', '#FF8C00', '#FF0000']
bars = plt.bar(CLASS_NAMES, counts, color=colors, edgecolor='black', alpha=0.8)

for i, bar in enumerate(bars):
    yval = bar.get_height()
    label_text = f"{int(yval)}\n(Sim: {avg_similarities[i]:.1f}%)"
    plt.text(
        bar.get_x() + bar.get_width()/2.0, 
        yval + (max(counts) * 0.02), 
        label_text, 
        ha='center', 
        va='bottom', 
        fontsize=9, 
        fontweight='bold'
    )

plt.xlabel("Assigned Keplerian Class (Data-Driven Density Spectrum)")
plt.ylabel("Number of Stars Assigned")
plt.title(f"Data-Anchored Iterative Sorting ({TOTAL_PASSES} Passes - Tier 1 Map Complete)")
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.ylim(0, max(counts) * 1.18)

print("Opening output plot window.")
plt.show()
