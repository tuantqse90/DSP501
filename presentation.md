# DSP501 — Presentation Slides

<!-- Slide deck in markdown format. Convert to PDF/PPTX with Marp, reveal.js, or similar. -->
<!-- Theme: NullShift dark (#0a0a0a background, #00ff41 accent, #00ffff headings) -->

---

## Slide 1: Title

# Environmental Sound Classification
### Does DSP Preprocessing Improve AI Performance?

**DSP501 — Digital Signal Processing Final Project**

Dataset: UrbanSound8K (8,732 clips, 10 classes)

March 2026

---

## Slide 2: Research Question

### Research Question

> Does applying classical DSP preprocessing to audio signals improve the accuracy of environmental sound classification?

**Two Pipelines:**

| Pipeline A (Baseline) | Pipeline B (DSP-Enhanced) |
|:---------------------:|:-------------------------:|
| Raw Audio | Raw Audio |
| ↓ | ↓ FIR Bandpass (50–10kHz) |
| Feature Extraction | ↓ Pre-emphasis (α=0.97) |
| ↓ | ↓ Normalization |
| Classification | ↓ Feature Extraction |
| | ↓ Classification |

---

## Slide 3: Dataset — UrbanSound8K

### UrbanSound8K Dataset

- **8,732** labeled clips, **10** environmental sound classes
- Max duration: 4 seconds, resampled to **22,050 Hz**
- **10 predefined folds** for cross-validation (never shuffled)

| Class | N | Type |
|-------|---|------|
| air_conditioner | 1,000 | Stationary |
| car_horn | 429 | Impulsive |
| children_playing | 1,000 | Stationary-like |
| dog_bark | 1,000 | Impulsive |
| drilling | 1,000 | Non-stationary |
| engine_idling | 1,000 | Stationary |
| gun_shot | 374 | Impulsive |
| jackhammer | 1,000 | Stationary-like |
| siren | 929 | Stationary-like |
| street_music | 1,000 | Stationary-like |

> *Figure: `fig_waveforms_per_class.png`, `fig_class_distribution.png`*

---

## Slide 4: Signal Analysis

### Signal Characterization

**Frequency Analysis:**
- FFT magnitude spectrum reveals distinct spectral profiles per class
- PSD shows energy distribution differences
- Spectral leakage analysis guides window selection

**Stationarity:**
- Stationary: air_conditioner, engine_idling (stable spectrum over time)
- Non-stationary: car_horn, dog_bark, gun_shot (impulsive events)

> *Figures: `fig_fft_per_class.png`, `fig_psd_per_class.png`*

---

## Slide 5: DSP Pipeline Design

### FIR Bandpass Filter

$$h[n] = h_d[n] \cdot w_{\text{Hann}}[n], \quad M = 101 \text{ taps}$$

- Passband: **50 Hz – 10,000 Hz**
- Window: Hann → smooth transition, -3dB at cutoffs
- **Linear phase** → preserves temporal structure
- Zero-phase implementation via `filtfilt`

### Why FIR over IIR?

| | FIR | IIR (Butterworth) |
|--|-----|-------------------|
| Phase | **Linear** | Non-linear |
| Stability | Always stable | Must verify poles |
| Order | 101 | 5 |

> *Figures: `fig_fir_frequency_response.png`, `fig_fir_vs_iir_comparison.png`*

---

## Slide 6: Pre-emphasis & Normalization

### Pre-emphasis Filter

$$x_p[n] = x[n] - 0.97 \cdot x[n-1]$$

- First-order high-pass filter: $H(z) = 1 - 0.97z^{-1}$
- Boosts high-frequency energy
- Compensates for spectral tilt in speech/environmental sounds

### Peak Normalization

$$x_{\text{out}}[n] = \frac{x_p[n]}{\max |x_p[n]|}$$

### SNR Improvement

| Class | ΔSNR |
|-------|------|
| children_playing | +4.5 dB |
| jackhammer | +2.9 dB |

> *Figures: `fig_before_after_children_playing.png`, `fig_snr_improvement.png`*

---

## Slide 7: Feature Engineering

### 931-Dimensional Feature Vector

| Feature | Dims |
|---------|------|
| MFCC (40) + Δ + ΔΔ → 7 stats each | 840 |
| Spectral (centroid, BW, rolloff, flatness, ZCR, RMS) → 7 stats | 42 |
| Spectral contrast (7 bands) → 7 stats | 49 |
| **Total** | **931** |

### Mel Spectrogram (CNN input)
- 128 mel bands × 173 time frames
- $f_{\min}=50$ Hz, $f_{\max}=10,000$ Hz

> *Figures: `fig_tsne_features.png`, `fig_feature_importance.png`*

---

## Slide 8: Models

### Three Classification Models

**1. SVM** (Support Vector Machine)
- RBF kernel, C=10, γ=scale
- PCA: 931 → 200 dimensions

**2. Random Forest**
- 500 trees, unlimited depth
- Full 931 features

**3. CNN-2D** (Convolutional Neural Network)
- 4 conv blocks: 32→64→128→256 channels
- BatchNorm + MaxPool + AdaptiveAvgPool
- Adam optimizer, early stopping
- Input: mel spectrogram (128×173)

---

## Slide 9: Evaluation Methodology

### 10-Fold Cross-Validation

- UrbanSound8K **predefined folds** (never shuffled)
- Each fold = test set once, remaining 9 = training
- Metrics: Accuracy, F1 (macro), 95% CI

### Statistical Tests

| Test | Purpose |
|------|---------|
| **Paired t-test** | Is mean difference ≠ 0? |
| **Wilcoxon signed-rank** | Non-parametric alternative |
| **Cohen's d** | Effect size magnitude |

Significance threshold: **α = 0.05**

---

## Slide 10: Results

### Classification Accuracy (10-Fold CV)

| Model | Pipeline A | Pipeline B | Δ | p-value |
|-------|:---------:|:---------:|:-:|:-------:|
| SVM | 70.1 ± 3.2% | 70.0 ± 3.6% | −0.12% | 0.846 |
| **Random Forest** | **71.5 ± 2.6%** | 71.2 ± 2.2% | −0.26% | 0.768 |
| CNN-2D | 66.7 ± 5.2% | 67.6 ± 4.8% | +0.94% | 0.625 |

### Effect Sizes

| Model | Cohen's d | Interpretation |
|-------|:---------:|:--------------:|
| SVM | 0.063 | **Negligible** |
| RF | 0.096 | **Negligible** |
| CNN | −0.160 | **Negligible** |

> *Figures: `fig_accuracy_comparison_barplot.png`, `fig_fold_accuracy_boxplot.png`*

---

## Slide 11: Statistical Analysis

### All p-values > 0.05 → No significant difference

```
SVM:  p = 0.846  →  NOT significant
RF:   p = 0.768  →  NOT significant
CNN:  p = 0.625  →  NOT significant
```

### Per-Fold Accuracy Shows High Variance

- CNN has widest spread (53.6% – 81.1%)
- RF is most stable (62.7% – 78.5%)
- Fold composition matters more than DSP

> *Figure: `fig_per_fold_accuracy.png`*

---

## Slide 12: Why Doesn't DSP Help?

### Key Insight

Modern feature extraction **already performs implicit DSP**:

1. **MFCC** = mel filterbank (bandpass) + cepstral transform
   → Already filters to perceptually relevant bands

2. **Mel spectrogram** uses $f_{\min}=50$, $f_{\max}=10,000$
   → Same passband as our FIR filter

3. **Statistical aggregation** (mean, std, skew, kurtosis)
   → Robust to noise — similar to normalization

4. **StandardScaler** in ML pipeline
   → Normalizes amplitude features

**The feature extraction pipeline already does what DSP aims to do.**

---

## Slide 13: Model Comparison

### Classical ML > Deep Learning (on small dataset)

| | RF | SVM | CNN-2D |
|--|:--:|:---:|:------:|
| Accuracy | **71.5%** | 70.1% | 66.7% |
| Variance | Low (2.6%) | Medium (3.2%) | High (5.2%) |
| Training | Fast (seconds) | Medium (minutes) | Slow (hours) |

**Why CNN underperforms:**
- ~7,800 training samples per fold → too few for deep learning
- No data augmentation
- Early stopping limits convergence

---

## Slide 14: Conclusion

### Answer to Research Question

> **DSP preprocessing does NOT significantly improve classification.**

### Key Takeaways

1. All p-values > 0.05, all effect sizes negligible
2. Feature extraction (MFCC, mel spectrogram) **already captures** what DSP provides
3. DSP does improve raw signal quality (SNR ↑) but doesn't translate to better accuracy
4. Random Forest (71.5%) > SVM (70.1%) > CNN-2D (66.7%)
5. Classical ML outperforms deep learning on small datasets

### Future Work
- Data augmentation for CNN
- Class-specific DSP (different filters per sound type)
- Pre-trained audio models (VGGish, PANNs)
- Attention mechanisms for temporal modeling

---

## Slide 15: Thank You

### DSP501 — Environmental Sound Classification

**Pipeline A ≈ Pipeline B** (p > 0.05)

*"Sometimes the best preprocessing is no preprocessing."*

---

### Technical Details

| Parameter | Value |
|-----------|-------|
| Sample rate | 22,050 Hz |
| Duration | 4 seconds |
| FIR order | 101 taps |
| Passband | 50–10,000 Hz |
| Features | 931 dims |
| Mel bands | 128 |
| CNN layers | 4 conv blocks |
| Folds | 10 (predefined) |
| Total figures | 42 |

Code: `src/` | Notebooks: `notebooks/01–06`
