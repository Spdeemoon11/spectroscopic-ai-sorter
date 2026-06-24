import numpy as np
import os

tier2_path = "tier2_line_metrics.npy"

# Fallback check if it's sitting inside your subfolder
if not os.path.exists(tier2_path):
    tier2_path = os.path.join("gaia_data_sort", "tier2_line_metrics.npy")

if not os.path.exists(tier2_path):
    print(f"❌ Error: Cannot find '{tier2_path}' anywhere!")
else:
    metrics = np.load(tier2_path)
    print("==========================================================")
    print("            TIER 2 OUTPUT MATRIX DIAGNOSTIC               ")
    print("==========================================================")
    print(f"Matrix Shape: {metrics.shape}")
    print(f"Total Stars Processed: {metrics.shape[0]}")
    print(f"Columns Recorded per Star: {metrics.shape[1]}")
    print("----------------------------------------------------------\n")
    
    # Specific columns Tier 3 is currently pulling from
    target_cols = {
        "h_alpha (Col 2)": metrics[:, 2],
        "h_beta (Col 4)": metrics[:, 4],
        "h_epsilon (Col 10)": metrics[:, 10],
        "he_i (Col 12)": metrics[:, 12],
        "he_ii (Col 14)": metrics[:, 14],
        "ca_h (Col 16)": metrics[:, 16],
        "ca_k (Col 18)": metrics[:, 18]
    }
    
    for name, data in target_cols.items():
        print(f"🔍 {name}:")
        print(f"   -> Min Value:  {np.min(data):.4f}")
        print(f"   -> Max Value:  {np.max(data):.4f}")
        print(f"   -> Mean Value: {np.mean(data):.4f}")
        print(f"   -> Std Dev:    {np.std(data):.4f}")
        print("-" * 40)