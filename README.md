# Benford's Law and Computational Hierarchy - Reproducibility Data

[中文版](README_CN.md) | English

## Overview

This repository contains the data and code for reproducing the robustness analysis in our study on Benford's Law and computational hierarchy in cellular automata. The analysis demonstrates that Rule 110's Benford-consistent behavior is robust across 1,000 independent runs with varied initial conditions.

## System Requirements

- **Python Version**: 3.13.0
- **Operating System**: Cross-platform (tested on Windows, should work on Linux/macOS)

## Repository Contents

### 1. `main.py`
The main analysis script that performs level-k predictability (L_k) analysis on cellular automata.

**Key Features:**
- Evolves four elementary cellular automata (Rules 32, 110, 30, 184) from random initial conditions
- Uses block entropy to identify active regions and determine dynamic height
- Computes level-k predictability (L_k) for k=1 to 30 using a dilated neighborhood predictor
- Evaluates first-digit distributions and performs Benford's law conformity tests
- Outputs detailed results for each run

**Main Steps:**
1. Generate 8,000-step CA evolution from random initial conditions
2. Apply block entropy screening to exclude static/steady-state regions
3. Sample 240 fixed-size windows (256×256) and compute L_k at each scale
4. Select the peak scale (largest range, k≠1) for Benford analysis
5. Calculate chi-square statistic, p-value, and MAD (Mean Absolute Deviation)

### 2. `requirements.txt`
Python package dependencies required to run the analysis.

**Core Dependencies:**
- `numpy==2.3.3` - Numerical computing
- `pandas==2.3.3` - Data manipulation and CSV I/O
- `matplotlib==3.10.6` - Visualization
- `scipy==1.16.2` - Statistical tests (chi-square)
- `seaborn==0.13.2` - Enhanced plotting

**Installation:**
```bash
pip install -r requirements.txt
```

### 3. `summary_results.csv`
Aggregated results from 1,000 independent runs for both Rule 110 and Rule 184.

**Columns:**
- `folder_name`: Timestamp identifier for each run (format: YYMMDD_HHMMSS)
- `dynamic_height`: Active height determined by block entropy for that run
- `mad_184`: Mean Absolute Deviation for Rule 184 at its peak scale
- `p_value_184`: Chi-square test p-value for Rule 184
- `mad_110`: Mean Absolute Deviation for Rule 110 at its peak scale
- `p_value_110`: Chi-square test p-value for Rule 110

**Key Statistics (from 1,000 runs):**
- Rule 110: Median MAD = 0.032, demonstrating consistent Benford conformity
- Rule 184: Median MAD = 0.113, showing systematic deviation from Benford's law

### 4. `运行结果/` (Results Directory)
Contains detailed outputs from all 1,000 independent runs.

**Directory Structure:**
Each subdirectory is named with a timestamp (e.g., `251019_201644/`) and contains:
- `*.csv` files: L_k values for each rule at all levels (k=1-30)
- `*.png` files: Visualization plots (boxplots, first-digit histograms)
- `*.txt` files: Run metadata (random seeds, parameters, statistics)

**Total Content:**
- ~7,000 files (4,000 CSVs, 2,000 PNGs, 1,000 TXTs)
- Each run is fully reproducible using the recorded random seeds

## Usage

### Running a Single Analysis

```bash
python main.py
```

This will:
1. Generate random seeds for reproducibility
2. Evolve all four CA rules
3. Compute L_k and perform Benford analysis
4. Save results as CSV files and generate plots

### Analyzing Existing Results

The `summary_results.csv` file can be loaded directly for statistical analysis:

```python
import pandas as pd
import numpy as np

# Load summary
df = pd.read_csv('summary_results.csv')

# Compute statistics
print(f"Rule 110 MAD: median={np.median(df['mad_110']):.4f}")
print(f"Rule 184 MAD: median={np.median(df['mad_184']):.4f}")
```

## Reproducibility Notes

- All runs use different random seeds, recorded in each output directory
- Block entropy parameters are fixed: block_height=40, ngram=3×3, bins=20, cutoff=1, consecutive=5
- Sampling parameters: 240 regions, 256×256 windows, 60% train / 40% test split
- The analysis is computationally intensive (~35 hours for 1,000 runs on standard hardware)

## License

This data is provided for research and reproducibility purposes. Please cite our paper if you use this code or data in your research.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: The directory names and file paths use Chinese characters (`运行结果` = "run results"). This is intentional and does not affect functionality on modern systems.

