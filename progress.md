# Progress — DSP501 Environmental Sound Classification

## Current Phase: Phase 6 — Report & Presentation (Complete)

### Sprint Log

| Date | Task | Status | Files Changed | Notes |
|------|------|--------|---------------|-------|
| 2026-03-13 | Project initialized | Done | All files | Full scaffold by Claude |
| 2026-03-13 | Dataset downloaded | Done | data/UrbanSound8K/ | 8732 samples, 10 folds, 10 classes |
| 2026-03-13 | Notebook 01: Signal analysis | Done | results/figures/ (17 figs) | Stationarity + frequency characterization complete |
| 2026-03-13 | Notebook 02: DSP pipeline | Done | results/figures/ (28 figs) | FIR/IIR design, before/after comparison |
| 2026-03-13 | Notebook 03: Feature engineering | Done | results/figures/ (39 figs) | 931-dim feature vector, t-SNE, importance analysis |
| 2026-03-14 | Feature cache extraction | Done | results/feature_cache.pkl (2.3GB) | All 8732 samples: raw+DSP features + mel specs |
| 2026-03-14 | Pipeline A: Classical ML | Done | results/pipeline_a_results.pkl | SVM=70.1%, RF=71.5% (10-fold CV) |
| 2026-03-14 | Pipeline B: Classical ML | Done | results/pipeline_b_results.pkl | SVM=70.0%, RF=71.2% (10-fold CV) |
| 2026-03-14 | Pipeline A: CNN-2D | Done | results/pipeline_a_results.pkl | CNN=66.7% (10-fold CV, MPS) |
| 2026-03-14 | Pipeline B: CNN-2D | Done | results/pipeline_b_results.pkl | CNN=67.6% (10-fold CV, MPS) |
| 2026-03-14 | Comparative analysis | Done | results/figures/ (42 figs), results/tables/ | Statistical tests, box plots, bar charts |
| 2026-03-14 | Final report | Done | report.md | Full technical report with math, tables, analysis |
| 2026-03-14 | Presentation slides | Done | presentation.md | 15 slides, ready for Marp/reveal.js |

### Decision Log

| Date | Decision | Context | Alternatives Considered |
|------|----------|---------|------------------------|
| 2026-03-13 | PyTorch over TensorFlow | Better MPS support on macOS, cleaner API | TensorFlow/Keras |
| 2026-03-13 | FIR as primary filter | Linear phase preserves temporal structure | IIR only |
| 2026-03-13 | 22050 Hz sample rate | Standard for audio ML, Nyquist covers useful range | 44100 Hz |
| 2026-03-13 | 931-dim feature vector | 120 MFCC stats + 42 spectral stats + 49 contrast stats | Fewer features |
| 2026-03-14 | PCA(200) for SVM | 931 dims too slow for RBF kernel SVM | Full 931 dims |
| 2026-03-14 | Feature caching | Re-extracting per fold too slow (~14h) | Per-fold extraction |

### Experiment Results

| Model | Pipeline A (Raw) | Pipeline B (DSP) | Δ (B−A) | p-value | Significant? |
|-------|-----------------|------------------|---------|---------|-------------|
| SVM | 70.1 ± 3.2% | 70.0 ± 3.6% | -0.12% | 0.8464 | No |
| Random Forest | 71.5 ± 2.6% | 71.2 ± 2.2% | -0.26% | 0.7680 | No |
| CNN-2D | 66.7 ± 5.2% | 67.6 ± 4.8% | +0.94% | 0.6252 | No |

### Key Findings
- **DSP preprocessing does NOT significantly improve classification** (p > 0.05 for all models)
- All effect sizes are negligible (Cohen's d < 0.2)
- Random Forest is the best-performing model (71.5% raw, 71.2% DSP)
- CNN-2D shows highest variance across folds (5.2% CI)
- CNN slightly benefits from DSP (+0.94%) but not statistically significant
- Classical ML (RF) outperforms deep learning (CNN-2D) on this dataset size
- Stationary-like: air_conditioner, engine_idling, children_playing, jackhammer, siren, street_music
- Non-stationary: car_horn, dog_bark, gun_shot, drilling
- FIR filter stable, IIR Butterworth stable (all poles inside unit circle)
- SNR improvement from DSP: +4.5dB (children_playing), +2.9dB (jackhammer)
- Class imbalance: car_horn (429), gun_shot (374) vs 1000 for most classes

### Blockers
_None currently_

### Upcoming
- [x] Write final report → `report.md`
- [x] Prepare presentation → `presentation.md`
- [ ] Convert presentation to PDF/PPTX (optional)
- [ ] Peer review / polish
