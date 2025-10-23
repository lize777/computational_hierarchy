# Level-k Predictability and Benford's Law Analysis

[English](README.md) | [中文](README_zh.md)

## Overview

This repository contains the experimental code and data for analyzing the relationship between computational hierarchy (measured by level-k predictability) and Benford's law conformity in cellular automata.

## Repository Structure
.
├── main.py # Main analysis script
├── summary_results.csv # Aggregated results from 1000 independent runs
└── 运行结果/ # Individual run results (1000 folders)
├── 251019_201644/
├── 251020_200551/
└── ...

## File Descriptions

### `main.py`

The main analysis script that:

1. **Generates CA space-time diagrams** for four elementary rules:
   - Rule 110 (Class IV, computationally universal)
   - Rule 30 (Class III, chaotic)
   - Rule 32 (Class I, convergent)
   - Rule 184 (Class II, periodic transport)

2. **Determines active height** using block entropy with consecutive low-entropy criterion on Rule 110, then applies the same cutoff to all rules for fair comparison.

3. **Computes level-k predictability (L_k)** for k=1 to 30:
   - L_k is defined as the odds-ratio lift of a local predictor
   - Uses a dilated 3-cell neighborhood at spacing k to forecast k steps ahead
   - L_k > 1 indicates a reducible scale (predictable structure)
   - L_k ≈ 1 indicates no structure (boundary-dependent or irreducible)

4. **Evaluates Benford's law conformity** at peak scales:
   - Selects the scale with largest L_k range (excluding k=1)
   - Computes chi-square statistic, p-value, and mean absolute deviation (MAD)
   - Lower MAD indicates stronger conformity to Benford's law

5. **Generates visualizations**:
   - Boxplots showing L_k distribution across scales for each rule
   - First-digit frequency histograms compared to Benford's theoretical distribution

### `summary_results.csv`

Aggregated statistics from 1000 independent runs with randomized initial conditions. Each row represents one run with the following columns:

- `folder_name`: Timestamp identifier for the run
- `dynamic_height`: Active height determined by block entropy (varies per run)
- `mad_184`: Mean absolute deviation for Rule 184 at its peak scale
- `p_value_184`: P-value for Rule 184
- `mad_110`: Mean absolute deviation for Rule 110 at its peak scale
- `p_value_110`: P-value for Rule 110

**Key findings**:
- Rule 110 consistently shows low MAD (median ≈ 0.032), indicating strong Benford conformity
- Rule 184 shows high MAD (median ≈ 0.113), indicating poor Benford conformity
- Results are robust across varied active heights (1000-8000 steps)

### `运行结果/` (Run Results)

Contains 1000 subdirectories, each storing the detailed results of one independent run, including:
- CSV files with L_k values for all levels and rules
- Generated plots (boxplots and Benford analysis)
- Metadata about block entropy and active height determination

## Usage

```bash
python main.py
```

**Parameters** (configurable in script):
- `BLOCK_HEIGHT = 40`: Height of entropy analysis blocks
- `NGRAM_SIZE = 3`: N-gram window size (3×3)
- `NUM_INTERVALS = 20`: Number of entropy bins
- `CUTOFF_INTERVAL = 1`: Entropy threshold (bin 1 = lowest entropy)
- `CONSECUTIVE_COUNT = 5`: Number of consecutive low-entropy blocks required

## Requirements
- Parameters (configurable in script):
- BLOCK_HEIGHT = 40: Height of entropy analysis blocks
- NGRAM_SIZE = 3: N-gram window size (3×3)
- NUM_INTERVALS = 20: Number of entropy bins
- CUTOFF_INTERVAL = 1: Entropy threshold (bin 1 = lowest entropy)
- CONSECUTIVE_COUNT = 5: Number of consecutive low-entropy blocks required
