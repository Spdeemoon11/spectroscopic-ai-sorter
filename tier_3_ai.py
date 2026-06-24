import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
MKK_CLASSES = ['I', 'II', 'III', 'IV', 'V']
REFERENCE_CSV_PATH = "reference.csv"
X_DATA_PATH = "X_spectra_ready.npy"
TIER2_DATA_PATH = "tier2_line_metrics.npy"

# Automatically apply directory fallback structure safely upfront
if not os.path.exists(X_DATA_PATH):
    X_DATA_PATH = os.path.join("gaia_data_sort", "X_spectra_ready.npy")

# ==========================================
# 1. LOAD DATA MATRIX & PIPELINE HANDSHAKES
# ==========================================
print("Loading preprocessed matrix and Tier 2 tracking metrics...")

if not os.path.exists(X_DATA_PATH):
    raise FileNotFoundError(f"Could not find spectral matrix at: {X_DATA_PATH}")
    
if not os.path.exists(TIER2_DATA_PATH):
    raise FileNotFoundError("Pipeline broken. Tier 2 metrics must exist before running Tier 3.")

X = np.load(X_DATA_PATH)
num_stars = X.shape[0]

# Load the atomic line configurations processed by Tier 2
tier2_metrics = np.load(TIER2_DATA_PATH) 

# ==========================================
# 2. LOAD PHYSICAL REFERENCE TEMPLATES
# ==========================================
print(f"\nLoading physical flux templates from {REFERENCE_CSV_PATH}...")
if not os.path.exists(REFERENCE_CSV_PATH):
    raise FileNotFoundError(f"Please create and populate '{REFERENCE_CSV_PATH}'!")

ref_df = pd.read_csv(REFERENCE_CSV_PATH)

# Target physical columns from your reference file
line_templates = ref_df[['h_alpha', 'h_beta', 'h_epsilon', 'he_i', 'he_ii', 'ca_h', 'ca_k']].values

# Isolate corresponding raw feature arrays out of tier2_line_metrics
t2_features = np.zeros((num_stars, 7))
t2_features[:, 0] = tier2_metrics[:, 2]   # h_alpha
t2_features[:, 1] = tier2_metrics[:, 4]   # h_beta
t2_features[:, 2] = tier2_metrics[:, 10]  # h_epsilon
t2_features[:, 3] = tier2_metrics[:, 12]  # he_i
t2_features[:, 4] = tier2_metrics[:, 14]  # he_ii
t2_features[:, 5] = tier2_metrics[:, 16]  # ca_h
t2_features[:, 6] = tier2_metrics[:, 18]  # ca_k

# ==========================================
# 3. HIGH-PRECISION COSINE SIMILARITY MATCHING
# ==========================================
print("Executing vector-normalized Cosine Similarity sorting sweep...")

# Step A: Compute the Euclidean norm of each template row and normalize them
template_norms = np.linalg.norm(line_templates, axis=1, keepdims=True)
template_norms[template_norms == 0] = 1.0  # Prevent division by zero
normalized_templates = line_templates / template_norms

# Step B: Compute the Euclidean norm of each star feature vector and normalize them
star_norms = np.linalg.norm(t2_features, axis=1, keepdims=True)
star_norms[star_norms == 0] = 1.0  # Prevent division by zero
normalized_stars = t2_features / star_norms

# Step C: Calculate true pattern alignment angles using matrix dot products
scores = np.zeros((num_stars, 5))
for idx in range(5):
    scores[:, idx] = np.dot(normalized_stars, normalized_templates[idx])

# Assign stars based on the absolute highest pattern match angle
final_assignments = np.argmax(scores, axis=1)

# ==========================================
# 4. SIMILARITY METRIC COMPILATION
# ==========================================
# Cosine similarity values fall between 0.0 and 1.0; scale to 100% directly
similarities = np.max(scores, axis=1) * 100

counts = []
avg_similarities = []

for idx in range(5):
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
print("\n[EXPORT] Publishing Tier 3 personal versioned data model...")
np.save("tier3_results.npy", final_assignments)
print("🎉 Success! Published normalized 'tier3_results.npy' with length bias correction.")

# ==========================================
# 6. HISTOGRAM RENDER STEP
# ==========================================
plt.figure(figsize=(12, 6))
colors = ['#FF4500', '#FF8C00', '#FFA500', '#FFD700', '#FFFF00']
bars = plt.bar(MKK_CLASSES, counts, color=colors, edgecolor='black', alpha=0.8)

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

plt.xlabel("MKK Luminosity Class (Cosine-Normalized Vector Match)")
plt.ylabel("Number of Stars Assigned")
plt.title("Pattern-Locked MKK Luminosity Sorting (Tier 3 Core Matrix Engine)")
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.ylim(0, max(counts) * 1.18)

print("Displaying updated stellar census distribution.")
plt.show()
