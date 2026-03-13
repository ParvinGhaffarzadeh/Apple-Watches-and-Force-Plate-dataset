# Apple Watch and Force Plate Dataset
Multi-modal dataset for ground reaction force estimation using consumer wearable sensors.

## 📊 Dataset
**Full dataset permanently archived on Zenodo:**
- Concept DOI: [10.5281/zenodo.17376716](https://doi.org/10.5281/zenodo.17376716)
- Version DOI (v1.0): [10.5281/zenodo.17376717](https://doi.org/10.5281/zenodo.17376717)
- 492 validated trials (395 triad-complete: wrist + waist + force plate)
- Five locomotor activities: walking, jogging, running, heel drop, step drop
- 10 healthy adults (aged 26–41 years)
- Force plate ground truth (1000 Hz) + Apple Watch IMU (~100 Hz)
- Wrist-worn (left wrist) and waist-worn (anterior waist, inferior to the navel) placements
Note: For trials with multiple force plate files (participants may contact 1–3 plates), the file with the most data rows is selected as the primary contact, corresponding to the plate with the largest ground reaction force signal.

## 📓 Repository Contents
- `FP_AW_Synchronised.ipynb` — Synchronisation and three-phase correlation validation pipeline
- `monte_carlo_sensitivity.py` — Monte Carlo timing sensitivity analysis (±10 ms perturbation)
- `trial_analysis.py` — Trial inventory, manifest generation, and QC pipeline
- Analysis scripts for ICC, CV%, and Pearson r QC metrics
- Preprocessing and visualisation utilities
- `alignment_log.csv` — Per-trial alignment log: estimated lag (ms), alignment method, overlap coverage, cross-correlation quality, and peak-time difference (ms)

## 🗂️ Dataset Structure
```
Dataset/
├── trial_manifest.csv          # Primary manifest (492 trials, UUID-based)
├── trial_manifest_triad.csv    # Triad-complete subset (395 trials)
├── data_dictionary.csv         # Machine-readable variable definitions
├── speed_data.xlsx             # Timing-gate speed records (n=183 valid trials)
├── alignment_log.csv           # Per-trial temporal alignment QC
└── [Participant folders]       # Raw and processed IMU + force plate CSV files
```

## 🚀 Quick Start
```python
import pandas as pd

# Load triad-complete manifest
manifest = pd.read_csv('trial_manifest.csv')
triad = manifest[manifest['triad_complete'] == True]  # n=395

# Recommended split for ML (avoid data leakage)
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(triad, groups=triad['participant']):
    pass  # participant-stratified folds
```

## ⚠️ ML Usage Guidelines
To prevent common misuse of this dataset in downstream modelling:

1. **Always use participant-level splits** (LOPO / GroupKFold) — never random row-wise splits
2. **Report per-participant metrics** (mean ± SD, CIs across folds), not only pooled performance
3. **Select the appropriate manifest subset** before splitting (triad-complete n=395 vs. full n=492)
4. **Treat impact tasks separately** — check saturation flags and prefer raw signals for high-frequency features
5. **Document missing-modality handling** — ensure missingness patterns do not differ between train and test partitions

## 📄 License
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

All materials in this repository (code, data, documentation) are released under the
**Creative Commons Attribution 4.0 International (CC-BY-4.0)** license.

You are free to:
- Share and redistribute the material
- Adapt, remix, and build upon the material
- Use for any purpose, including commercially

**Attribution requirement:** Please cite the paper and dataset:
```bibtex
@article{ghaffarzadeh2025dataset,
  author    = {Ghaffarzadeh, Parvin and Chakraborty, Debarati and
               Aslansefat, Koorosh and Dostan, Ali and Papadopoulos, Yiannis},
  title     = {A Multi-Modal Dataset for Ground Reaction Force Estimation
               Using Consumer Wearable Sensors},
  journal   = {Scientific Data},
  year      = {2025},
  note      = {Under review},
  doi       = {10.5281/zenodo.17376717}
}
```

## 📖 Citation
If you use this dataset or code in your research, please cite:

> Ghaffarzadeh, P., Chakraborty, D., Aslansefat, K., Dostan, A., & Papadopoulos, Y. (2025).
> A Multi-Modal Dataset for Ground Reaction Force Estimation Using Consumer Wearable Sensors.
> *Scientific Data* (under review). https://doi.org/10.5281/zenodo.17376717

## 🔗 Links
- **Dataset (Zenodo)**: [https://doi.org/10.5281/zenodo.17376717](https://doi.org/10.5281/zenodo.17376717)
- **Paper**: Submitted to *Scientific Data* (under review)
- **Contact**: p.ghaffarzadeh@hull.ac.uk

## 📝 About
This dataset supports research in:
- Wearable biomechanics and locomotion analysis
- Ground reaction force estimation from consumer devices
- Machine learning model development and benchmarking
- Wrist vs. waist sensor placement studies
- Transfer learning and domain adaptation for biomechanical inference
- Consumer device validation against laboratory-grade equipment

Developed at the Department of Artificial Intelligence and Modelling,
University of Hull, UK.
