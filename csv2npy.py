import pandas as pd
import numpy as np
import ast

# Load full dataset
df = pd.read_csv("gaia_data_sort/starset1.csv", comment="#")
print(f"Processing {len(df)} stars...")

# Pre-allocate array based on first row length
num_bins = len(ast.literal_eval(df.iloc[0]["flux"]))
X_data = np.zeros((len(df), 1, num_bins), dtype=np.float32)

window_size = 2
half_w = window_size // 2

for row in range(len(df)):
    flux = np.array(ast.literal_eval(df.iloc[row]["flux"]), dtype=np.float32)
    flux_error = np.array(ast.literal_eval(df.iloc[row]["flux_error"]), dtype=np.float32)
    
    # Normalize proportionally
    max_val = np.max(flux)
    flux_norm = flux / max_val
    error_norm = flux_error / max_val
    
    # Error-weighted softening
    weights = 1.0 / (error_norm**2 + 1e-8)
    flux_softened = np.copy(flux_norm)
    for i in range(half_w, len(flux_norm) - half_w):
        window_flux = flux_norm[i - half_w : i + half_w + 1]
        window_weights = weights[i - half_w : i + half_w + 1]
        flux_softened[i] = np.sum(window_flux * window_weights) / np.sum(window_weights)
        
    X_data[row, 0, :] = flux_softened

# Save to a lightning-fast, compressed binary file
np.save("X_spectra_ready.npy", X_data)
print("Data saved successfully as 'X_spectra_ready.npy'!")