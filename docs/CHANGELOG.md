# Changelog — DSP501

## [0.0.1] — 2026-03-13

### Added
- Project scaffolded with full directory structure
- `config.py` — Central configuration with all hyperparameters
- `src/data_loader.py` — UrbanSound8K dataset loading and fold management
- `src/signal_analysis.py` — Time/frequency/time-frequency analysis functions
- `src/dsp_pipeline.py` — FIR/IIR filter design and preprocessing pipeline
- `src/feature_extraction.py` — MFCC, spectral features, mel spectrogram extraction
- `src/models/classical_ml.py` — SVM and Random Forest with GridSearchCV
- `src/models/deep_learning.py` — CNN-2D and CNN-1D with PyTorch
- `src/evaluation.py` — Metrics, confusion matrix, ROC, statistical tests
- `src/visualization.py` — All plotting functions (NullShift dark theme)
- 6 Jupyter notebooks (01-06) covering full experiment pipeline
- Full documentation suite (README, ARCHITECTURE, TECH_STACK, DEV_GUIDE, etc.)
- CLAUDE.md for Claude Code CLI integration
- progress.md for sprint tracking
