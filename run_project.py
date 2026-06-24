#!/usr/bin/env python3
# Copyright (c) 2026 Vrishank Yadav
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Master pipeline script to execute the spectroscopic sorting sequence.
Ensures correct ordering of the 4 cascade AI models.
"""

import subprocess
import sys

def run_script(script_name):
    print(f"==================================================")
    print(f"Executing: {script_name}")
    print(f"==================================================")
    
    # Runs the sub-script and waits for it to finish before moving to the next
    result = subprocess.run([sys.executable, script_name], capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with exit code {result.returncode}. Halting pipeline.")
        sys.exit(result.returncode)
    print(f"SUCCESS: {script_name} completed running.\n")

def main():
    # Define your cryptic file names here in the exact order they must run
    pipeline_steps = [
        "gaia_data_sort/csv2npy.py",             # Step 0 : Turn Gaia DR3 data csv into numpy file for ai processing
        "gaia_data_sort/tier_1_ai.py",           # Step 1 : Harvard classes (O, B, A, F, G, K, M)
        "gaia_data_sort/tier_2_ai.py",           # Step 2 : Subclasses and Morgan-Keenan luminosity
        "gaia_data_sort/tier2output_check.py",   # Step 2a: Check if the tier 2 output is correctly formatted or not
        "gaia_data_sort/tier_3_ai.py",            # Step 3 : Luminousity Classes identifier
        "gaia_data_sort/tier_4_ai.py",            # Step 4 : Final verification / anomaly rechecker
        "gaia_data_sort/image_txt_generation.py" # Step 5 : Make images and metadata
    ]
    
    print("Starting Spectroscopic AI Sorting Pipeline...")
    for step in pipeline_steps:
        run_script(step)
    print("Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
