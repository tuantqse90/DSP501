# Dev Guide — DSP501

> Version: 1.0 | Last updated: 2026-03-13

## Local Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download UrbanSound8K
# Go to: https://urbansounddataset.weebly.com/urbansound8k.html
# Extract to: data/UrbanSound8K/

# 4. Verify structure
ls data/UrbanSound8K/audio/fold1/  # Should list .wav files
ls data/UrbanSound8K/metadata/     # Should have UrbanSound8K.csv

# 5. Run first notebook
cd notebooks && jupyter notebook 01_signal_analysis.ipynb
```

## Coding Conventions

- **Config**: All hyperparameters in `config.py` — no magic numbers
- **Type hints**: Use throughout (`def func(x: np.ndarray) -> dict:`)
- **Docstrings**: Google style for all public functions
- **Imports**: stdlib → third-party → local, separated by blank lines
- **Naming**: `snake_case` for functions/variables, `UPPER_CASE` for constants
- **Line length**: 100 characters max

## Git Workflow

```bash
# Branch naming
feature/signal-analysis
feature/dsp-pipeline
feature/cnn-model
fix/filter-stability

# Commit format (Conventional Commits)
feat: add FIR bandpass filter design
fix: correct MFCC delta computation
docs: update architecture diagram
test: add evaluation metrics tests
```

## Notebook Execution Order

Notebooks MUST be run in order — each depends on previous outputs:

1. `01_signal_analysis.ipynb` — Understand the data
2. `02_dsp_pipeline.ipynb` — Design and validate filters
3. `03_feature_engineering.ipynb` — Extract and analyze features
4. `04_pipeline_a_raw.ipynb` — Baseline experiments
5. `05_pipeline_b_dsp.ipynb` — DSP experiments
6. `06_comparative_analysis.ipynb` — Statistical comparison

## Related Docs

- [TECH_STACK.md](TECH_STACK.md) — Dependencies
- [TESTING_PLAN.md](TESTING_PLAN.md) — Testing strategy
