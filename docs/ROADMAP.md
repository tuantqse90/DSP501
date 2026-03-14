# Roadmap — DSP501

> Version: 1.0 | Last updated: 2026-03-13

## Phase 1: Setup & Signal Analysis (Week 1)

- [x] Project scaffold and documentation
- [ ] Download and verify UrbanSound8K dataset
- [ ] Run notebook 01: Raw signal exploration
- [ ] Characterize all 10 classes (time, frequency, time-frequency)
- [ ] Identify stationary vs non-stationary signals
- [ ] Generate signal analysis figures

## Phase 2: DSP Pipeline (Week 2)

- [ ] Run notebook 02: Filter design
- [ ] Implement and analyze FIR bandpass filter
- [ ] Implement and analyze IIR Butterworth filter
- [ ] Compare FIR vs IIR (frequency response, phase, stability)
- [ ] Apply preprocessing to dataset samples
- [ ] Generate before/after comparison figures

## Phase 3: Feature Engineering (Week 2-3)

- [ ] Run notebook 03: Feature extraction
- [ ] Extract MFCC + delta + delta-delta
- [ ] Extract spectral features (centroid, bandwidth, rolloff, contrast, flatness)
- [ ] Statistical aggregation for fixed-length vectors
- [ ] Generate mel spectrograms for CNN
- [ ] Feature importance analysis (RF)
- [ ] t-SNE visualization

## Phase 4: Pipeline A Experiments (Week 3)

- [ ] Run notebook 04: Baseline with raw audio
- [ ] Train SVM with GridSearchCV (10-fold CV)
- [ ] Train Random Forest with tuning (10-fold CV)
- [ ] Train CNN-2D on raw mel spectrograms (10-fold CV)
- [ ] Record all fold-wise metrics
- [ ] Save Pipeline A results

## Phase 5: Pipeline B Experiments (Week 3-4)

- [ ] Run notebook 05: With DSP preprocessing
- [ ] Train SVM on DSP-processed features (10-fold CV)
- [ ] Train Random Forest on DSP-processed features (10-fold CV)
- [ ] Train CNN-2D on DSP-processed mel spectrograms (10-fold CV)
- [ ] Record all fold-wise metrics
- [ ] Save Pipeline B results

## Phase 6: Comparative Analysis (Week 4)

- [ ] Run notebook 06: Statistical comparison
- [ ] Generate comparison table
- [ ] Paired t-test and Wilcoxon test
- [ ] Cohen's d effect size
- [ ] Confusion matrices for all model-pipeline combinations
- [ ] ROC curves and AUC

## Phase 7: Report & Presentation (Week 4-5)

- [ ] Write IEEE format report (10-12 pages)
- [ ] Include all figures and tables
- [ ] Address all 6 discussion questions
- [ ] Prepare 12-15 min presentation slides

## Related Docs

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [MODEL_SPEC.md](MODEL_SPEC.md)
