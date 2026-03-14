# DSP501 — Environmental Sound Classification (UrbanSound8K)
## Claude CLI Prompt: Full Project Scaffold

---

## 🎯 PROJECT OVERVIEW

Build a complete **Environmental Sound Classification** system for the DSP501 Final Group Project. The project uses the **UrbanSound8K** dataset and compares two pipelines:

- **Pipeline A**: Raw Signal → AI Model (minimal preprocessing)
- **Pipeline B**: DSP Preprocessing → Feature Extraction → AI Model

The goal is to scientifically evaluate whether DSP preprocessing improves classification performance.

---

## 📁 PROJECT STRUCTURE

```
dsp501-env-sound/
├── README.md
├── requirements.txt
├── config.py                    # All hyperparameters, paths, random seeds
├── data/
│   └── UrbanSound8K/           # Dataset (download separately)
│       ├── audio/
│       │   ├── fold1/ ... fold10/
│       └── metadata/
│           └── UrbanSound8K.csv
├── notebooks/
│   ├── 01_signal_analysis.ipynb        # Raw signal exploration
│   ├── 02_dsp_pipeline.ipynb           # Filter design & frequency analysis
│   ├── 03_feature_engineering.ipynb    # Feature extraction & visualization
│   ├── 04_pipeline_a_raw.ipynb         # Pipeline A experiments
│   ├── 05_pipeline_b_dsp.ipynb         # Pipeline B experiments
│   └── 06_comparative_analysis.ipynb   # Statistical comparison
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading, fold management
│   ├── signal_analysis.py       # Waveform, FFT, PSD analysis
│   ├── dsp_pipeline.py          # Filter design (FIR/IIR), preprocessing
│   ├── feature_extraction.py    # MFCC, spectral features, statistical
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical_ml.py      # SVM, Random Forest
│   │   └── deep_learning.py     # CNN (1D & 2D), optional RNN
│   ├── evaluation.py            # Metrics, confusion matrix, ROC, CI
│   └── visualization.py         # All plotting functions
├── results/
│   ├── figures/
│   └── tables/
└── report/
    └── DSP501_Report.tex        # IEEE format LaTeX template
```

---

## 📊 DATASET

**UrbanSound8K** — https://urbansounddataset.weizIR.net/urbansound8k.html

- **10 classes**: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
- **8,732 labeled audio clips** (≤4 seconds each)
- **10 pre-defined folds** for cross-validation
- **Sampling rate**: varies (mostly 22,050 Hz or 44,100 Hz) — resample all to **22,050 Hz**
- **Citation**: J. Salamon, C. Jacoby, and J.P. Bello, "A Dataset and Taxonomy for Urban Sound Research," ACM Multimedia, 2014.

⚠️ **IMPORTANT**: UrbanSound8K requires using the **predefined 10 folds** — do NOT shuffle data across folds. Use 10-fold cross-validation with the given splits.

---

## 🔧 IMPLEMENTATION DETAILS

### 1. config.py — Central Configuration

```python
"""
Central configuration for DSP501 Environmental Sound Classification project.
All hyperparameters, paths, and random seeds defined here for reproducibility.
"""

import os

# === Reproducibility ===
RANDOM_SEED = 42
N_FOLDS = 10  # UrbanSound8K predefined folds

# === Paths ===
DATA_DIR = "data/UrbanSound8K"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
METADATA_PATH = os.path.join(DATA_DIR, "metadata/UrbanSound8K.csv")
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# === Audio Parameters ===
TARGET_SR = 22050          # Resample all audio to this rate
AUDIO_DURATION = 4.0       # Max duration in seconds
N_SAMPLES = int(TARGET_SR * AUDIO_DURATION)  # 88,200 samples

# === DSP Parameters ===
# Bandpass filter (Pipeline B)
FILTER_TYPE = "bandpass"
FILTER_LOW_FREQ = 50       # Hz — remove DC offset and very low noise
FILTER_HIGH_FREQ = 10000   # Hz — remove high-freq noise above useful range
FIR_ORDER = 101            # FIR filter order (odd number)
IIR_ORDER = 5              # Butterworth IIR order

# === Frequency Analysis ===
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
WINDOW_TYPE = "hann"

# === Feature Extraction (Pipeline B) ===
N_MFCC = 40
N_MELS = 128
FMIN = 50
FMAX = 10000

# === ML Hyperparameters ===
# SVM
SVM_C_RANGE = [0.1, 1, 10, 100]
SVM_GAMMA_RANGE = ['scale', 'auto', 0.01, 0.001]
SVM_KERNEL = 'rbf'

# Random Forest
RF_N_ESTIMATORS = [100, 200, 500]
RF_MAX_DEPTH = [10, 20, 50, None]

# CNN
CNN_EPOCHS = 100
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
CNN_PATIENCE = 10  # Early stopping

# === Classes ===
CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]
```

---

### 2. src/data_loader.py

```
Module responsibilities:
- Load UrbanSound8K metadata CSV
- Load audio files using librosa, resample to TARGET_SR
- Pad/truncate to fixed length (N_SAMPLES)
- Return fold-based splits (respect predefined folds)
- Class distribution analysis and visualization
- Handle corrupted/missing files gracefully

Key functions:
- load_metadata() → DataFrame
- load_audio(file_path, sr=TARGET_SR) → np.array
- get_fold_split(fold_id) → (train_files, test_files)
- analyze_class_distribution() → dict, plot
```

---

### 3. src/signal_analysis.py — Raw Signal Characterization

**This is critical for the report's "Signal Analysis" section.**

```
Analyze raw UrbanSound8K signals:

1. TIME-DOMAIN ANALYSIS:
   - Waveform visualization (sample from each class)
   - Amplitude statistics: mean, std, RMS, peak, crest factor
   - Zero-crossing rate per class
   - Signal duration distribution

2. FREQUENCY-DOMAIN ANALYSIS:
   - FFT magnitude spectrum per class
   - Power Spectral Density (PSD) using Welch's method
   - Dominant frequency bands per class
   - Bandwidth analysis

3. TIME-FREQUENCY ANALYSIS:
   - STFT spectrograms (linear and log scale)
   - Mel spectrograms
   - Compare window sizes: 512, 1024, 2048, 4096
   - Discuss time-frequency resolution trade-off

4. NOISE ANALYSIS:
   - Identify noise sources in urban recordings
   - Estimate SNR where possible
   - Background noise characterization

5. SIGNAL CHARACTERIZATION:
   - Classify signals as stationary vs non-stationary
   - Justify with examples:
     * Stationary-like: air_conditioner, engine_idling
     * Non-stationary: car_horn, gun_shot, siren
   - Spectral leakage analysis with different windows (rectangular, Hann, Hamming, Blackman)

Key plots to generate (save to results/figures/):
- fig_waveforms_per_class.png
- fig_fft_per_class.png
- fig_psd_per_class.png
- fig_spectrogram_comparison.png
- fig_window_size_comparison.png
- fig_spectral_leakage.png
- fig_class_duration_distribution.png
```

---

### 4. src/dsp_pipeline.py — Digital Filter Design (Pipeline B)

```
Implement and analyze digital filters for preprocessing:

1. FIR BANDPASS FILTER (Primary):
   - Design: scipy.signal.firwin, order=101
   - Passband: 50 Hz – 10,000 Hz
   - Window method (Hann window)
   - Analysis:
     * Frequency response (magnitude in dB)
     * Phase response
     * Group delay
     * Impulse response
   - Apply using scipy.signal.filtfilt (zero-phase)

2. IIR BUTTERWORTH FILTER (For comparison):
   - Design: scipy.signal.butter, order=5
   - Same passband: 50 Hz – 10,000 Hz
   - Analysis:
     * Frequency response
     * Phase response
     * Pole-zero plot
     * Stability verification (all poles inside unit circle)
   - Apply using scipy.signal.filtfilt

3. ADDITIONAL PREPROCESSING:
   - Pre-emphasis filter: y[n] = x[n] - 0.97 * x[n-1]
   - Amplitude normalization (peak normalization to [-1, 1])
   - Silence removal (trim leading/trailing silence, threshold-based)

4. BEFORE vs AFTER COMPARISON:
   For multiple samples per class, generate:
   - Waveform: raw vs filtered
   - FFT spectrum: raw vs filtered
   - Spectrogram: raw vs filtered
   - PSD: raw vs filtered
   - SNR improvement estimation

5. FIR vs IIR COMPARISON:
   - Compare frequency responses
   - Compare phase responses (linear phase advantage of FIR)
   - Compare computational cost
   - Justify final filter choice

Key plots:
- fig_fir_frequency_response.png
- fig_fir_phase_response.png
- fig_iir_frequency_response.png
- fig_iir_pole_zero.png
- fig_fir_vs_iir_comparison.png
- fig_before_after_waveform.png
- fig_before_after_spectrum.png
- fig_before_after_spectrogram.png
- fig_snr_improvement.png
```

---

### 5. src/feature_extraction.py — Feature Engineering (Pipeline B)

```
Extract and justify features for classification:

1. SPECTRAL FEATURES (from librosa):
   - MFCC (40 coefficients) + delta + delta-delta → 120 features
     * Mathematical explanation: DCT of log mel-spectrogram
     * Why MFCC: models human auditory perception
   - Spectral Centroid: "center of mass" of spectrum
   - Spectral Bandwidth: spread around centroid
   - Spectral Rolloff: frequency below which 85% energy concentrated
   - Spectral Contrast: valley-to-peak ratio in sub-bands
   - Spectral Flatness: tonality measure (noise-like vs tonal)
   - Zero-Crossing Rate
   - RMS Energy

2. TEMPORAL FEATURES:
   - Temporal centroid
   - Attack time / onset strength
   - Envelope statistics

3. STATISTICAL AGGREGATION:
   For each frame-level feature, compute over time:
   - Mean, Std, Min, Max, Median
   - Skewness, Kurtosis
   → Creates fixed-length feature vector per audio clip

4. MEL SPECTROGRAM (for CNN input):
   - 128 mel bands
   - Convert to dB scale
   - Normalize per sample (zero mean, unit variance)
   - Shape: (128, T) where T = ceil(N_SAMPLES / HOP_LENGTH)

5. FEATURE ANALYSIS:
   - Feature correlation heatmap
   - Feature importance (from Random Forest)
   - t-SNE / PCA visualization of feature space
   - Per-class feature distributions (box plots for top features)

Key functions:
- extract_handcrafted_features(audio, sr) → np.array (fixed-length vector)
- extract_mel_spectrogram(audio, sr) → np.array (2D)
- extract_mfcc_features(audio, sr) → np.array
- extract_all_features(audio_list, sr) → DataFrame

Key plots:
- fig_feature_correlation.png
- fig_feature_importance.png
- fig_tsne_features.png
- fig_mfcc_per_class.png
- fig_mel_spectrogram_per_class.png
```

---

### 6. src/models/ — AI/ML Models

#### 6.1 classical_ml.py

```
Implement classical ML models:

1. SVM (Support Vector Machine):
   - Kernel: RBF
   - Hyperparameter tuning: GridSearchCV over C and gamma
   - Feature scaling: StandardScaler (fit on train, transform test)
   - Input: handcrafted feature vectors

2. RANDOM FOREST:
   - Hyperparameter tuning: n_estimators, max_depth
   - Feature importance extraction
   - Input: handcrafted feature vectors

Both models must be evaluated with:
- Pipeline A input: statistical features from RAW mel-spectrogram (no DSP filtering)
- Pipeline B input: handcrafted features from DSP-processed audio

Cross-validation: Use UrbanSound8K's 10 predefined folds
- Train on 9 folds, test on 1, rotate
- Report mean ± 95% CI for all metrics
```

#### 6.2 deep_learning.py

```
Implement deep learning models using PyTorch or TensorFlow/Keras:

1. CNN-2D (Primary DL model):
   Architecture:
   - Input: Mel spectrogram (1 x 128 x T)
   - Conv2D(32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
   - Conv2D(64, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
   - Conv2D(128, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
   - Conv2D(256, 3x3) → BatchNorm → ReLU → GlobalAvgPool
   - Dropout(0.5)
   - Dense(128) → ReLU → Dropout(0.3)
   - Dense(10) → Softmax
   
   Training:
   - Optimizer: Adam (lr=0.001)
   - Loss: CrossEntropyLoss
   - Early stopping (patience=10)
   - Learning rate scheduler (ReduceLROnPlateau)
   - Data augmentation (optional): time shift, pitch shift, add noise

2. CNN-1D (Optional secondary DL model):
   - Input: Raw waveform (Pipeline A) or filtered waveform (Pipeline B)
   - 1D convolutions to learn directly from waveform

Both CNN models evaluated with:
- Pipeline A: Mel spectrogram from RAW audio
- Pipeline B: Mel spectrogram from DSP-processed audio

Must include:
- Training/validation loss curves
- Overfitting analysis (train vs val accuracy gap)
- Model parameter count
- Inference time comparison
```

---

### 7. src/evaluation.py — Metrics & Statistical Analysis

```
Comprehensive evaluation module:

1. PERFORMANCE METRICS (per fold and aggregated):
   - Accuracy
   - Precision (macro, weighted)
   - Recall (macro, weighted)
   - F1-score (macro, weighted)
   - Per-class metrics

2. VISUALIZATION:
   - Confusion matrix (normalized, with annotations)
   - ROC curve (one-vs-rest, per class)
   - AUC scores
   - Per-class accuracy bar chart

3. STATISTICAL COMPARISON (Pipeline A vs B):
   - Mean ± 95% Confidence Interval across 10 folds
   - Paired t-test (or Wilcoxon signed-rank test)
   - Effect size (Cohen's d)
   - Box plot of fold-wise accuracy distributions

4. RESULTS TABLE:
   Generate a comparison table:
   
   | Model          | Pipeline | Accuracy      | F1 (macro)    | p-value |
   |----------------|----------|---------------|---------------|---------|
   | SVM            | A (Raw)  | XX.X ± X.X%  | XX.X ± X.X%  | —       |
   | SVM            | B (DSP)  | XX.X ± X.X%  | XX.X ± X.X%  | 0.XXX   |
   | Random Forest  | A (Raw)  | XX.X ± X.X%  | XX.X ± X.X%  | —       |
   | Random Forest  | B (DSP)  | XX.X ± X.X%  | XX.X ± X.X%  | 0.XXX   |
   | CNN-2D         | A (Raw)  | XX.X ± X.X%  | XX.X ± X.X%  | —       |
   | CNN-2D         | B (DSP)  | XX.X ± X.X%  | XX.X ± X.X%  | 0.XXX   |

Key plots:
- fig_confusion_matrix_[model]_[pipeline].png
- fig_roc_curve_[model]_[pipeline].png
- fig_accuracy_comparison_barplot.png
- fig_fold_accuracy_boxplot.png
- fig_training_curves_cnn.png
```

---

### 8. Required Discussion Points

Address these in the report's Discussion section with evidence:

```
1. DOES DSP PREPROCESSING IMPROVE PERFORMANCE?
   - Compare Pipeline A vs B metrics
   - Which classes benefit most from filtering?
   - Hypothesis: filtering helps noisy classes (drilling, jackhammer)
     but may hurt tonal classes (siren) if passband too narrow

2. DISCRIMINATIVE FREQUENCY BANDS:
   - Show PSD per class → identify where classes differ
   - air_conditioner/engine_idling: low-freq dominated
   - car_horn/siren: mid-freq tonal components
   - gun_shot: broadband impulse
   - children_playing/street_music: wide spectral spread

3. DOES FILTERING REMOVE USEFUL INFORMATION?
   - Compare feature distributions before/after
   - Check if any class accuracy drops after filtering
   - Discuss passband selection trade-off

4. EFFECT ON OVERFITTING:
   - Compare train-test gap for Pipeline A vs B
   - DSP may act as regularization by removing noise
   - Fewer irrelevant features → less overfitting

5. COMPUTATIONAL COMPLEXITY:
   - Measure preprocessing time per sample
   - Feature extraction time
   - Model training time
   - Total pipeline time comparison

6. IS DSP NECESSARY WITH DEEP LEARNING?
   - CNNs can learn filters from data
   - But DSP provides domain knowledge as inductive bias
   - Smaller dataset → DSP more beneficial
   - Compare CNN performance on raw vs processed
```

---

### 9. Report Template (IEEE Format)

```
Structure for 10-12 page IEEE single-column report:

1. INTRODUCTION (1 page)
   - Problem statement: urban sound classification
   - Motivation: noise monitoring, smart cities, surveillance
   - Research question: Does DSP preprocessing improve ESC?
   - Contributions summary

2. SIGNAL ANALYSIS (1.5 pages)
   - UrbanSound8K dataset description
   - Signal characteristics per class
   - Stationarity analysis
   - Noise source identification
   - Sampling rate justification (Nyquist)

3. DSP METHODOLOGY (2 pages)
   - Filter design (FIR/IIR)
   - Frequency response analysis
   - Phase response analysis
   - Pre-emphasis filter
   - Window function selection
   - Before/after signal comparison

4. FEATURE ENGINEERING (1.5 pages)
   - MFCC derivation and justification
   - Spectral features with formulas
   - Statistical aggregation
   - Feature selection/importance

5. AI MODELING (1.5 pages)
   - SVM architecture and tuning
   - Random Forest architecture and tuning
   - CNN-2D architecture and training
   - Hyperparameter tables

6. EXPERIMENTAL RESULTS (1.5 pages)
   - Pipeline A results (all models)
   - Pipeline B results (all models)
   - Confusion matrices
   - ROC curves

7. COMPARATIVE ANALYSIS (1 page)
   - Performance comparison table
   - Statistical significance tests
   - Signal-level evaluation (SNR, spectrum)

8. DISCUSSION (1 page)
   - Address all 6 required discussion questions
   - Critical analysis with evidence

9. LIMITATIONS (0.5 page)
   - Dataset limitations (4-sec clips, class imbalance)
   - Filter design choices
   - Computational constraints
   - Generalization concerns

10. CONCLUSION (0.5 page)
    - Summary of findings
    - Answer to research question
    - Future work

ETHICS STATEMENT (as required):
- UrbanSound8K is publicly available under CC BY-NC 3.0
- No human subjects involved in data collection by our team
- All code properly cited
- AI tools usage disclosed

REFERENCES (minimum 10):
- Salamon et al., 2014 (UrbanSound8K)
- Piczak, 2015 (ESC benchmarks)
- Davis & Mermelstein, 1980 (MFCC)
- Oppenheim & Schafer (DSP textbook)
- Additional DSP and ML references
```

---

### 10. requirements.txt

```
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
soundfile>=0.12.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0
torchaudio>=2.0.0
tqdm>=4.65.0
jupyter>=1.0.0
```

---

## ⚡ EXECUTION ORDER

```
Step 1: Setup environment, download UrbanSound8K, verify data loading
Step 2: Run signal analysis (notebook 01) — understand the data
Step 3: Design and implement DSP pipeline (notebook 02) — filter design
Step 4: Extract features (notebook 03) — both raw and DSP-processed
Step 5: Train Pipeline A models (notebook 04) — baseline
Step 6: Train Pipeline B models (notebook 05) — with DSP
Step 7: Comparative analysis (notebook 06) — statistical tests
Step 8: Generate all figures and tables for report
Step 9: Write report in IEEE format
Step 10: Prepare presentation slides (12-15 min)
```

---

## 🎓 KEY TIPS FOR HIGH MARKS

1. **Theoretical justification everywhere** — Don't just apply filters; explain WHY you chose those parameters
2. **Visualize everything** — Before/after comparisons are gold
3. **Statistical rigor** — Mean ± CI, paired tests, not just single accuracy numbers
4. **Critical discussion** — Acknowledge when DSP doesn't help and explain why
5. **Respect UrbanSound8K folds** — Never shuffle across folds
6. **Mathematical formulas** — Include equations for MFCC, spectral features, filter design
7. **Reproducibility** — Random seeds, documented hyperparameters, clear README
