# DSP501 Final Report — Environmental Sound Classification

**Does DSP Preprocessing Improve AI-Based Environmental Sound Classification?**

> Course: DSP501 — Digital Signal Processing
> Date: March 2026
> Dataset: UrbanSound8K (Salamon et al., 2014)

---

## Abstract

This project investigates whether traditional DSP preprocessing improves AI-based environmental sound classification. We design and compare two pipelines on the UrbanSound8K dataset (8,732 clips, 10 classes): **Pipeline A** feeds raw audio directly to classifiers, while **Pipeline B** applies FIR bandpass filtering, pre-emphasis, and amplitude normalization before classification. Three models are evaluated — SVM, Random Forest, and CNN-2D — using 10-fold cross-validation with the dataset's predefined folds. Our results show **no statistically significant difference** between the two pipelines (all p > 0.05, all Cohen's d < 0.2), suggesting that modern feature extraction methods (MFCCs, spectral features) already capture the relevant information, making explicit DSP preprocessing redundant for this task.

---

## 1. Introduction

### 1.1 Motivation

Environmental Sound Classification (ESC) is a fundamental problem in audio signal processing with applications in smart cities, surveillance, wildlife monitoring, and assistive technologies. A key question in the field is whether traditional DSP preprocessing — filtering, pre-emphasis, normalization — provides meaningful improvements when combined with modern machine learning methods.

### 1.2 Research Question

> **Does applying classical DSP preprocessing (bandpass filtering, pre-emphasis, normalization) to audio signals improve the accuracy of environmental sound classification compared to using raw audio directly?**

### 1.3 Approach

We design two experimental pipelines:

- **Pipeline A (Baseline)**: Raw audio → Feature Extraction → Classification
- **Pipeline B (DSP-Enhanced)**: Raw audio → FIR Bandpass Filter → Pre-emphasis → Normalization → Feature Extraction → Classification

Both pipelines are evaluated with identical models and evaluation methodology, differing only in whether DSP preprocessing is applied.

---

## 2. Dataset

### 2.1 UrbanSound8K

The UrbanSound8K dataset contains 8,732 labeled sound excerpts (≤ 4 seconds) of urban sounds across 10 classes:

| Class | Samples | Category |
|-------|---------|----------|
| air_conditioner | 1,000 | Stationary |
| car_horn | 429 | Non-stationary |
| children_playing | 1,000 | Stationary-like |
| dog_bark | 1,000 | Non-stationary |
| drilling | 1,000 | Non-stationary |
| engine_idling | 1,000 | Stationary |
| gun_shot | 374 | Non-stationary |
| jackhammer | 1,000 | Stationary-like |
| siren | 929 | Stationary-like |
| street_music | 1,000 | Stationary-like |

**Key properties:**
- 10 predefined folds for cross-validation (never shuffled)
- Class imbalance: car_horn (429) and gun_shot (374) are underrepresented
- Audio resampled to **22,050 Hz**, padded/truncated to **4 seconds** (88,200 samples)

### 2.2 Signal Characteristics

Stationarity analysis reveals two distinct groups:
- **Stationary-like signals** (relatively stable spectral content): air_conditioner, engine_idling, children_playing, jackhammer, siren, street_music
- **Non-stationary signals** (impulsive/transient): car_horn, dog_bark, gun_shot, drilling

This distinction is relevant because DSP filtering has different effects on stationary vs. non-stationary signals.

---

## 3. DSP Pipeline Design

### 3.1 FIR Bandpass Filter

**Design method**: Window method with Hann window

The ideal bandpass impulse response:

$$h_d[n] = \frac{\sin(\omega_h n)}{\pi n} - \frac{\sin(\omega_l n)}{\pi n}$$

where $\omega_l = 2\pi \cdot 50 / 22050$ and $\omega_h = 2\pi \cdot 10000 / 22050$.

**Parameters:**
- Passband: 50 Hz – 10,000 Hz
- Order: 101 taps
- Window: Hann
- Implementation: Zero-phase filtering via `scipy.signal.filtfilt`

**Properties:**
- Linear phase: $\phi(\omega) = -\frac{M-1}{2}\omega$ (preserves temporal structure)
- Constant group delay: $\tau_g = 50$ samples
- Always stable (FIR systems have no poles)

**Transfer function:**

$$H(z) = \sum_{k=0}^{100} h[k] \, z^{-k}$$

**Zero-phase implementation** (forward-backward filtering):

$$y[n] = \mathcal{F}^{-1}\left\{ |H(e^{j\omega})|^2 \cdot X(e^{j\omega}) \right\}$$

### 3.2 IIR Butterworth Filter (Comparison)

For comparison, we also implemented a 5th-order IIR Butterworth bandpass filter:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \left(\frac{\Omega}{\Omega_c}\right)^{2N}}$$

Converted to digital domain via bilinear transform with frequency pre-warping. The IIR filter is **stable** (all poles inside the unit circle) but has **nonlinear phase**, which distorts temporal structure.

**Decision**: FIR chosen as the primary filter because linear phase is critical for preserving the temporal characteristics of environmental sounds.

### 3.3 Pre-emphasis Filter

First-order high-pass filter to boost high frequencies:

$$x_p[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha = 0.97$$

Transfer function: $H_{\text{pre}}(z) = 1 - 0.97z^{-1}$

### 3.4 Peak Normalization

$$x_{\text{out}}[n] = \frac{x_p[n]}{\max_n |x_p[n]|}$$

### 3.5 Complete Pipeline B Processing Chain

$$x[n] \xrightarrow{\text{FIR bandpass}} x_f[n] \xrightarrow{\text{pre-emphasis}} x_p[n] \xrightarrow{\text{normalize}} x_{\text{out}}[n]$$

### 3.6 DSP Impact on Signal Quality

SNR improvement measurements show DSP preprocessing does improve signal quality:
- children_playing: +4.5 dB
- jackhammer: +2.9 dB
- Impulsive signals (car_horn, gun_shot): SNR estimation unreliable (no quiet frames)

---

## 4. Feature Engineering

### 4.1 Handcrafted Features (for SVM and Random Forest)

A 931-dimensional feature vector is extracted from each audio clip:

| Feature Group | Per-Frame Features | Statistics | Dimensions |
|--------------|-------------------|------------|------------|
| MFCC (40 coeffs) + Δ + ΔΔ | 120 | 7 (mean, std, min, max, median, skew, kurtosis) | 840 |
| Spectral centroid | 1 | 7 | 7 |
| Spectral bandwidth | 1 | 7 | 7 |
| Spectral rolloff | 1 | 7 | 7 |
| Spectral flatness | 1 | 7 | 7 |
| Zero-crossing rate | 1 | 7 | 7 |
| RMS energy | 1 | 7 | 7 |
| Spectral contrast (7 bands) | 7 | 7 | 49 |
| **Total** | | | **931** |

### 4.2 Mel Spectrogram (for CNN-2D)

- 128 mel bands, $f_{\min}=50$ Hz, $f_{\max}=10,000$ Hz
- FFT size: 2048, hop length: 512
- Output shape: (128, 173) per clip
- Converted to log scale (dB)

### 4.3 Dimensionality Reduction

For SVM, PCA was applied to reduce from 931 to 200 dimensions (SVM with RBF kernel scales as $O(n^2 d)$, making 931 features computationally prohibitive for ~7,800 training samples per fold).

---

## 5. Classification Models

### 5.1 SVM (Support Vector Machine)

- Kernel: RBF (Radial Basis Function)
- Preprocessing: StandardScaler → PCA(200)
- Hyperparameters: $C = 10$, $\gamma = \text{scale}$
- Decision function: $f(\mathbf{x}) = \text{sign}\left(\sum_i \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$

### 5.2 Random Forest

- 500 trees, unlimited depth
- Preprocessing: StandardScaler
- Full 931-dimensional feature vector (tree-based methods handle high dimensions well)
- Parallelized across all CPU cores

### 5.3 CNN-2D (Convolutional Neural Network)

Architecture:

```
Input: (1, 128, 173) — 1 channel × 128 mel bands × 173 time frames

Conv2d(1→32, 3×3)   → BatchNorm → ReLU → MaxPool(2)
Conv2d(32→64, 3×3)  → BatchNorm → ReLU → MaxPool(2)
Conv2d(64→128, 3×3) → BatchNorm → ReLU → MaxPool(2)
Conv2d(128→256, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(1)

Dropout(0.5) → Dense(256→128) → ReLU → Dropout(0.3) → Dense(128→10)
```

Training:
- Optimizer: Adam (lr = 0.001)
- Loss: CrossEntropyLoss
- LR Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early stopping: patience = 5 epochs
- Batch size: 64
- Device: Apple MPS (Metal Performance Shaders)

---

## 6. Experimental Methodology

### 6.1 Evaluation Protocol

- **10-fold cross-validation** using UrbanSound8K's predefined folds
- Each fold serves as the test set once; remaining 9 folds for training
- Folds are **never shuffled** — same samples always in the same fold (as per dataset guidelines)

### 6.2 Metrics

- **Accuracy**: Overall correct predictions / total predictions
- **F1 Score (macro)**: Harmonic mean of precision and recall, averaged across classes
- **95% Confidence Interval**: $\text{CI} = 1.96 \cdot \frac{\sigma}{\sqrt{n}}$ where $n = 10$ folds

### 6.3 Statistical Tests

To determine if the difference between Pipeline A and B is statistically significant:

1. **Paired t-test**: Tests if the mean difference across folds is significantly different from zero

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}, \quad \bar{d} = \frac{1}{n}\sum_{i=1}^{n}(a_i - b_i)$$

2. **Wilcoxon signed-rank test**: Non-parametric alternative (does not assume normal distribution of differences)

3. **Cohen's d (effect size)**: Measures the magnitude of the difference

$$d = \frac{\bar{d}}{s_d}$$

Interpretation: |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large

### 6.4 Feature Caching

To avoid re-extracting features for each fold (estimated ~14 hours), all features were pre-extracted and cached:
- `feature_cache.pkl` (2.3 GB): raw features (8732 × 931), DSP features (8732 × 931), raw mel specs (8732 × 128 × 173), DSP mel specs (8732 × 128 × 173), labels, fold assignments

---

## 7. Results

### 7.1 Classification Accuracy

| Model | Pipeline A (Raw) | Pipeline B (DSP) | Δ (B − A) | p-value (t-test) | Cohen's d |
|-------|:----------------:|:----------------:|:---------:|:-----------------:|:---------:|
| **SVM** | 70.1 ± 3.2% | 70.0 ± 3.6% | −0.12% | 0.8464 | 0.063 |
| **Random Forest** | **71.5 ± 2.6%** | 71.2 ± 2.2% | −0.26% | 0.7680 | 0.096 |
| **CNN-2D** | 66.7 ± 5.2% | 67.6 ± 4.8% | +0.94% | 0.6252 | −0.160 |

### 7.2 F1 Score (Macro)

| Model | Pipeline A (Raw) | Pipeline B (DSP) | Δ (B − A) | p-value |
|-------|:----------------:|:----------------:|:---------:|:-------:|
| SVM | 71.2 ± 3.1% | 71.2 ± 3.5% | −0.04% | 0.9553 |
| Random Forest | 72.8 ± 2.2% | 72.6 ± 1.9% | −0.21% | 0.7945 |
| CNN-2D | 66.7 ± 5.3% | 66.8 ± 5.3% | +0.08% | 0.9660 |

### 7.3 Statistical Significance

| Model | Paired t-test p | Wilcoxon p | Cohen's d | Effect Size |
|-------|:---------------:|:----------:|:---------:|:-----------:|
| SVM | 0.8464 | 0.5566 | 0.063 | Negligible |
| Random Forest | 0.7680 | 0.8457 | 0.096 | Negligible |
| CNN-2D | 0.6252 | 0.6953 | −0.160 | Negligible |

**None of the three models show a statistically significant difference** between Pipeline A and Pipeline B at the α = 0.05 significance level.

### 7.4 Per-Fold Accuracy Breakdown

The per-fold accuracy shows high variance, particularly for CNN-2D:

- **SVM range**: 63.4% – 78.6% (Pipeline A), 62.8% – 78.6% (Pipeline B)
- **RF range**: 62.7% – 78.5% (Pipeline A), 65.9% – 78.0% (Pipeline B)
- **CNN range**: 53.6% – 78.0% (Pipeline A), 54.6% – 81.1% (Pipeline B)

This fold-to-fold variance is expected given the heterogeneous nature of environmental sounds and class imbalance.

---

## 8. Analysis and Discussion

### 8.1 Why Doesn't DSP Help?

**Key insight**: Modern audio feature extraction methods already perform implicit DSP preprocessing.

1. **MFCC computation** applies mel-scale filterbanks (effectively bandpass filtering) and computes cepstral coefficients, already emphasizing perceptually relevant frequency bands.

2. **Mel spectrogram** uses parameters $f_{\min} = 50$ Hz and $f_{\max} = 10,000$ Hz, which is the same passband as our FIR filter. The mel spectrogram inherently ignores frequencies outside this range.

3. **Statistical aggregation** (mean, std, skewness, kurtosis over time) provides robustness against noise — similar to what normalization and filtering aim to achieve.

4. **StandardScaler** in the ML pipeline normalizes features to zero mean and unit variance, overlapping with the effect of amplitude normalization.

In essence, the feature extraction pipeline already captures what DSP preprocessing aims to provide, making the explicit DSP step redundant.

### 8.2 Model Comparison

- **Random Forest** is the best overall model (71.5%), benefiting from its ability to handle high-dimensional features (931 dims) without PCA.
- **SVM** achieves comparable results (70.1%) but required PCA dimensionality reduction for computational feasibility.
- **CNN-2D** underperforms classical ML (66.7%), likely due to limited dataset size (~7,800 training samples per fold) and the complexity of the 4-layer architecture.

### 8.3 CNN Performance

The CNN-2D shows the highest fold-to-fold variance (CI = 5.2%) and lower absolute accuracy than classical ML. This is expected for deep learning on small datasets:
- ~7,800 training samples is relatively small for a 4-layer CNN
- Early stopping (patience=5) may prevent full convergence on some folds
- Data augmentation (not implemented) could improve CNN performance

### 8.4 Limitations

1. **No data augmentation**: CNN performance could improve with time-shifting, pitch-shifting, noise injection
2. **Fixed SVM hyperparameters**: Grid search was skipped for computational reasons; optimal C/gamma could improve SVM results
3. **No per-class analysis**: Some classes may benefit from DSP more than others
4. **Single filter design**: Only one FIR configuration (50–10,000 Hz, order 101) was tested; different passbands might yield different results

---

## 9. Figures

All 42 figures are saved in `results/figures/`. Key figures include:

| Figure | Description |
|--------|-------------|
| `fig_waveforms_per_class.png` | Raw waveforms for all 10 classes |
| `fig_fft_per_class.png` | FFT magnitude spectrum per class |
| `fig_psd_per_class.png` | Power spectral density per class |
| `fig_fir_frequency_response.png` | FIR bandpass magnitude and phase response |
| `fig_iir_frequency_response.png` | IIR Butterworth response (comparison) |
| `fig_fir_vs_iir_comparison.png` | FIR vs IIR overlay |
| `fig_iir_pole_zero.png` | IIR pole-zero diagram (stability verification) |
| `fig_before_after_*.png` | Before/after DSP for each class (10 figures) |
| `fig_snr_improvement.png` | SNR improvement from DSP per class |
| `fig_tsne_features.png` | t-SNE visualization of 931-dim features |
| `fig_feature_importance.png` | Random Forest feature importance |
| `fig_feature_correlation.png` | Feature correlation heatmap |
| `fig_accuracy_comparison_barplot.png` | Pipeline A vs B accuracy bar chart |
| `fig_fold_accuracy_boxplot.png` | Box plot of fold-wise accuracy |
| `fig_per_fold_accuracy.png` | Per-fold line plots for all models |

---

## 10. Conclusion

### 10.1 Answer to Research Question

**DSP preprocessing does not significantly improve environmental sound classification accuracy.** Across all three models (SVM, Random Forest, CNN-2D), the differences between Pipeline A (raw) and Pipeline B (DSP-preprocessed) are:
- Not statistically significant (all p > 0.05)
- Negligible in effect size (all |Cohen's d| < 0.2)
- Practically insignificant (|Δ| < 1%)

### 10.2 Key Takeaways

1. Modern feature extraction (MFCCs, mel spectrograms) already implicitly performs the filtering and normalization that explicit DSP preprocessing provides.
2. Classical ML (Random Forest: 71.5%) outperforms deep learning (CNN-2D: 66.7%) on this dataset size.
3. The FIR bandpass filter does improve raw signal SNR (+2.9 to +4.5 dB for some classes), but this improvement does not translate to better classification accuracy.
4. DSP preprocessing is not wasted — understanding filter design, frequency analysis, and signal characteristics provides valuable engineering insight even when the classification improvement is negligible.

### 10.3 Future Work

- Apply data augmentation for CNN training
- Test class-specific DSP preprocessing (e.g., different filters for stationary vs. non-stationary sounds)
- Experiment with deeper/wider CNN architectures or pre-trained models (e.g., VGGish, PANNs)
- Investigate attention mechanisms for handling variable-length audio

---

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *ACM Multimedia*.
2. Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing*. Pearson.
3. Davis, S., & Mermelstein, P. (1980). Comparison of Parametric Representations for Monosyllabic Word Recognition. *IEEE TASSP*.
4. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE MLSP*.

---

## Appendix A: Reproducibility

### Environment
- Python 3.13.2
- PyTorch 2.6 (MPS backend)
- scikit-learn 1.6
- librosa 0.10
- scipy 1.15

### Random Seeds
- All experiments use `RANDOM_SEED = 42`
- Numpy, PyTorch, and scikit-learn seeds set before each fold

### Data
- UrbanSound8K v2 from Zenodo
- 10 predefined folds, never shuffled
- Audio resampled to 22,050 Hz, padded/truncated to 4s

### Code
All source code is in the `src/` directory. Notebooks in `notebooks/` reproduce the full pipeline from raw data to final results.
