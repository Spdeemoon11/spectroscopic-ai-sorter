import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
TIER2_DATA_PATH = "tier2_line_metrics.npy"
REFERENCE_CSV_PATH = "reference.csv"
TIER4_EXPORT_PATH = "tier4_final_census.csv"

SPECTRAL_CLASSES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
LUMINOSITY_CLASSES = ['I', 'II', 'III', 'IV', 'V']

# Adjust data path fallback correctly upfront
if not os.path.exists(TIER2_DATA_PATH):
    TIER2_DATA_PATH = os.path.join("gaia_data_sort", "tier2_line_metrics.npy")

# ==========================================
# 1. LOAD DATA MATRIX & PIPELINE HANDSHAKES
# ==========================================
print("Loading Tier 2 metrics and reference profiles...")
if not os.path.exists(TIER2_DATA_PATH):
    raise FileNotFoundError("Pipeline broken. Tier 2 metrics must exist before running Tier 4.")
if not os.path.exists(REFERENCE_CSV_PATH):
    raise FileNotFoundError(f"Missing '{REFERENCE_CSV_PATH}' for structural templates.")

tier2_metrics = np.load(TIER2_DATA_PATH)
num_stars = tier2_metrics.shape[0]

ref_df = pd.read_csv(REFERENCE_CSV_PATH)
line_templates = ref_df[['h_alpha', 'h_beta', 'h_epsilon', 'he_i', 'he_ii', 'ca_h', 'ca_k']].values

# Isolate feature arrays out of tier2_line_metrics
t2_features = np.zeros((num_stars, 7))
t2_features[:, 0] = tier2_metrics[:, 2]   # h_alpha
t2_features[:, 1] = tier2_metrics[:, 4]   # h_beta
t2_features[:, 2] = tier2_metrics[:, 10]  # h_epsilon
t2_features[:, 3] = tier2_metrics[:, 12]  # he_i
t2_features[:, 4] = tier2_metrics[:, 14]  # he_ii
t2_features[:, 5] = tier2_metrics[:, 16]  # ca_h
t2_features[:, 6] = tier2_metrics[:, 18]  # ca_k

# ==========================================
# 2. VECTORIZED INFRASTRUCTURE OPERATION
# ==========================================
print("\nProcessing multi-task forward evaluation pass...")

# --- Task A: Luminosity (Tier 3 Job via Cosine Similarity) ---
template_norms = np.linalg.norm(line_templates, axis=1, keepdims=True)
template_norms[template_norms == 0] = 1.0
normalized_templates = line_templates / template_norms

star_norms = np.linalg.norm(t2_features, axis=1, keepdims=True)
star_norms[star_norms == 0] = 1.0
normalized_stars = t2_features / star_norms

lum_scores = np.zeros((num_stars, 5))
for i in range(5):
    lum_scores[:, i] = np.dot(normalized_stars, normalized_templates[i])

final_luminosity_idx = np.argmax(lum_scores, axis=1)
lum_certainty = np.max(lum_scores, axis=1) * 100

# --- Task B: Spectral Sequence (Tier 1/2 Job via Ionization Density Mapping) ---
# Summing total line energy profiles to establish a core temperature index map
spectral_density = np.sum(t2_features, axis=1)
max_density = np.max(spectral_density) if np.max(spectral_density) > 0 else 1.0

# Evenly distribute temperature index thresholds across the density score space
spec_scores = np.zeros((num_stars, 7))
density_bins = np.linspace(0, max_density, 8)
for i in range(7):
    # Gaussian radial mapping function centered on each sequence bin
    center = (density_bins[i] + density_bins[i+1]) / 2.0
    spec_scores[:, i] = np.exp(-((spectral_density - center) ** 2) / (2 * (max_density / 7) ** 2))

final_spectral_idx = np.argmax(spec_scores, axis=1)
spec_certainty = np.max(spec_scores, axis=1) * 100

# Compute master fused certainty score per star
fused_certainty = (spec_certainty + lum_certainty) / 2.0

# ==========================================
# 3. MASTER UNIFIED CSV LEDGER EXPORT
# ==========================================
print(f"\n[EXPORT] Publishing integrated catalog to '{TIER4_EXPORT_PATH}'...")

output_df = pd.DataFrame({
    'star_index': np.arange(num_stars),
    'spectral_class_idx': final_spectral_idx,
    'spectral_class_name': [SPECTRAL_CLASSES[i] for i in final_spectral_idx],
    'luminosity_class_idx': final_luminosity_idx,
    'luminosity_class_name': [LUMINOSITY_CLASSES[i] for i in final_luminosity_idx],
    'fused_certainty_pct': fused_certainty
})

output_df.to_csv(TIER4_EXPORT_PATH, index=False)
print("🎉 Success! The master multi-tier catalog is complete and saved.")

# ==========================================
# 4. PLOT VISUALIZATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: Spectral Distribution
spec_counts = [np.sum(final_spectral_idx == i) for i in range(7)]
spec_colors = ['#0000FF', '#4B0082', '#8A2BE2', '#40E0D0', '#FFD700', '#FF8C00', '#FF0000']
ax1.bar(SPECTRAL_CLASSES, spec_counts, color=spec_colors, edgecolor='black', alpha=0.8)
ax1.set_title("Fused Spectral Class Distribution (Tier 1/2 Tasks)")
ax1.set_xlabel("Stellar Class")
ax1.set_ylabel("Star Count")
ax1.grid(axis='y', linestyle=':', alpha=0.5)

# Panel 2: Luminosity MKK Distribution
lum_counts = [np.sum(final_luminosity_idx == i) for i in range(5)]
lum_colors = ['#FF4500', '#FF8C00', '#FFA500', '#FFD700', '#FFFF00']
ax2.bar(LUMINOSITY_CLASSES, lum_counts, color=lum_colors, edgecolor='black', alpha=0.8)
ax2.set_title("Fused MKK Luminosity Distribution (Tier 3 Task)")
ax2.set_xlabel("MKK Luminosity Class")
ax2.set_ylabel("Star Count")
ax2.grid(axis='y', linestyle=':', alpha=0.5)

plt.suptitle(f"Unified Tier 4 Forward Inference Engine (Avg Certainty: {np.mean(fused_certainty):.1f}%)")
plt.tight_layout()
plt.show()