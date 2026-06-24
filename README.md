# Spectroscopic-AI-Sorter (aka SAS)
Automated spectroscopic pipeline for Gaia data, utilizing 4 cascading AI models to classify stellar Harvard classes, subclasses/MK, Balmer lines, and structural rechecking.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20829903.svg)](https://doi.org/10.5281/zenodo.20829903)

## Prerequisites to running this project- ##
Folder structure must look like-

```
 Main/
 |-- reference.csv
 |-- gaia_data_sort
   |-- (Star dataset in .csv format, named starset1)
   |-- csv2npy
   |--image_txt_generation.py
   |--...
```

> Warning- in the current code state, the pipline only works if you name the dataset ```starset1.csv```, to change this edit line:               
> ```df = pd.read_csv("gaia_data_sort/starset1.csv", comment="#")```
> in ```csv2npy.py```

## Running the code ##
To automate the process, run_project.py has been provided which will convert the .csv to .npy, run all the ai tiers in chronological order, and generate the image and metadata output in a folder in location-
```
Main/
 |-- reference.csv
 |-- zoouniverse_preview <—— the final script, image_txt_generation.py will make this folder and its subfolders listed below
    |-- images
        |-- star_XXXX_plain.png    <—— this shows the plain continum of the normailsed star flux
        |-- star_XXXX_balmer.png   <—— this highlights the spectroscopic features
        |-- ...
    |-- metadata
        |-- star_XXXX_metadata.txt <—— this shows the features, stellar subclass, mk class, and AI confidence for all the features
        |--...
 |-- gaia_data_sort
     |-- starset1.csv
     |-- csv2npy
     |--image_txt_generation.py
     |--...
```

## How It Works ##
The Spectroscopic AI Sorter (SAS) operates as a **cascading neural network pipeline**. Instead of using a single complex model to guess every stellar parameter at once, the pipeline splits the physics of spectral interpretation into specialized, sequential layers.
### 1. Data Transformation Layer (`csv2npy.py`)
* **The Physics:** Raw Gaia DR3 data provides flux values across specific wavelengths. 
* **The Algorithm:** This layer acts as a data pipeline pipeline. It strips out irrelevant catalog metadata, isolates the continuous raw spectral flux vectors, normalizes the intensity scales to prevent exposure bias, and serializes the matrix into a single unified NumPy tensor (`X_spectra_ready.npy`) for high-speed GPU ingestion.

### 2. Tier 1: Deep Continuum Classification (`tier_1_ai.py`)
* **The Physics:** A star's overall blackbody radiation curve shape dictates its broad temperature profile, spanning from ultra-hot Blue stars ($\sim 30,000\text{ K}$) to cool Red stars ($\sim 2,400\text{ K}$).
* **The Algorithm:** A convolutional neural network (CNN) scans the large-scale slope and shape of the continuum. It filters the input data into the 7 primary **Harvard Spectral Classes**: `O`, `B`, `A`, `F`, `G`, `K`, or `M`.

### 3. Tier 2: Micro-Feature Temperature Subclassification (`tier_2_ai.py`)
* **The Physics:** Within a broad class (like `G`), tiny variations in temperature alter the ionization and excitation states of atoms, changing the specific line depths of metals and hydrogen.
* **The Algorithm:** This layer takes the output classification from Tier 1 and passes it to a highly granular model. It focuses on narrow wavelength windows to calculate specific absorption line-depth ratios, assigning a precise numerical subclass from `0` to `9` (e.g., refining a broad `G` class down to a `G2`).

### 4. Tier 3: Pressure & Width Luminosity Determination (`tier_3_ai.py`)
* **The Physics:** The physical size of a star affects its atmospheric pressure. Giant stars have low-pressure atmospheres, producing incredibly sharp, narrow spectral lines. Dwarf stars (like our Sun) have high-pressure atmospheres, causing collisional broadening that creates wide, winged spectral lines.
* **The Algorithm:** This neural net ignores the overall slope and explicitly measures line widths, wing profiles, and pressure-sensitive lines. It outputs the standard **Morgan-Keenan (MK) Luminosity Class** (e.g., Supergiant `I`, Giant `III`, or Main-Sequence Dwarf `V`).

### 5. Tier 4: Convolutional Rechecking & Quality Control (`tier_4_ai.py`)
* **The Physics:** Real-world astronomical data contains instrumental noise, cosmic ray artifacts, and overlapping features that can confuse individual models.
* **The Algorithm:** This acts as an automated peer-review layer. It feeds the raw spectrum through a global multi-task verification network to evaluate if the combined results from Tiers 1, 2, and 3 are physically co-dependent and logical. 
  * If the confidence score passes the safety threshold, the parameters are approved.
  * If the network detects a contradiction (e.g., an unphysical combination of high-temperature features with low-pressure profiles), it flags the asset, routing it directly into a dedicated human-classification workflow on the **Zooniverse** project page.

### 6. Asset Export Pipeline (`image_txt_generation.py`)
* **The Process:** Validated classifications are mapped alongside their corresponding spectral graphs. The script automatically plots clean, labeled PNG images of the spectrum and pairs them with structured manifest metadata `.txt` text files, ready for seamless batch uploading to the Zooniverse project manager platform.
