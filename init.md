# Init Checklist — DSP501 Environmental Sound Classification

## Status: Done — Project Scaffolded

### Pre-requisites
- [ ] Python 3.10+ installed
- [ ] UrbanSound8K dataset downloaded
- [ ] pip / venv available

### Environment Setup
- [ ] Create virtual environment (`python -m venv venv`)
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Place UrbanSound8K in `data/UrbanSound8K/`
- [ ] Verify data: `ls data/UrbanSound8K/audio/fold1/`

### Documentation
- [x] All core docs generated (README, ARCHITECTURE, TECH_STACK, DEV_GUIDE, TESTING_PLAN, DEPLOYMENT, SECURITY, ROADMAP, CHANGELOG)
- [x] Conditional docs generated (MODEL_SPEC, WORKFLOW_SPEC)
- [x] CLAUDE.md configured
- [x] progress.md initialized

### Source Code
- [x] config.py — Central configuration
- [x] src/data_loader.py — Dataset loading
- [x] src/signal_analysis.py — Signal characterization
- [x] src/dsp_pipeline.py — FIR/IIR filter design
- [x] src/feature_extraction.py — Feature engineering
- [x] src/models/classical_ml.py — SVM, Random Forest
- [x] src/models/deep_learning.py — CNN-2D, CNN-1D
- [x] src/evaluation.py — Metrics & statistical tests
- [x] src/visualization.py — All plotting (NullShift theme)

### Notebooks
- [x] 01_signal_analysis.ipynb
- [x] 02_dsp_pipeline.ipynb
- [x] 03_feature_engineering.ipynb
- [x] 04_pipeline_a_raw.ipynb
- [x] 05_pipeline_b_dsp.ipynb
- [x] 06_comparative_analysis.ipynb

### First Tasks
- [ ] Download UrbanSound8K dataset
- [ ] Run notebook 01 (signal analysis)
- [ ] Run notebook 02 (DSP pipeline)
- [ ] Write first test (`tests/test_dsp_pipeline.py`)

### Verification
- [ ] All notebooks run without errors
- [ ] Figures generated in results/figures/
- [ ] Pipeline A and B results saved
- [ ] Report drafted
