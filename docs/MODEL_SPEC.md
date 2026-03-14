# Model Specification — DSP501

> Version: 1.0 | Last updated: 2026-03-13

## Models Overview

| Model | Type | Input | Tuning |
|-------|------|-------|--------|
| SVM | Classical ML | Handcrafted features | GridSearchCV (C, gamma) |
| Random Forest | Classical ML | Handcrafted features | GridSearchCV (n_estimators, max_depth) |
| CNN-2D | Deep Learning | Mel spectrogram (128 x T) | Adam + ReduceLROnPlateau |

## SVM

- **Kernel**: RBF
- **Feature scaling**: StandardScaler (fit on train, transform test)
- **Hyperparameter search**:
  - C: [0.1, 1, 10, 100]
  - gamma: ['scale', 'auto', 0.01, 0.001]
- **CV**: 3-fold within training set for grid search

## Random Forest

- **n_estimators**: [100, 200, 500]
- **max_depth**: [10, 20, 50, None]
- **Feature importance**: Extracted for analysis
- **n_jobs**: -1 (parallel)

## CNN-2D Architecture

```
Input: (1, 128, T)  — 1 channel, 128 mel bands, T time frames

Conv2d(1→32, 3x3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)
Conv2d(32→64, 3x3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)
Conv2d(64→128, 3x3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)
Conv2d(128→256, 3x3, pad=1) → BatchNorm2d → ReLU → AdaptiveAvgPool2d(1)

Dropout(0.5) → Dense(256→128) → ReLU → Dropout(0.3) → Dense(128→10)
```

### Training

- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Early stopping**: patience=10 epochs
- **Batch size**: 32
- **Max epochs**: 100

## CNN-1D Architecture (Optional)

```
Input: (1, 88200)  — raw/filtered waveform

Conv1d(1→32, kernel=80, stride=4)  → BatchNorm1d → ReLU → MaxPool1d(4)
Conv1d(32→64, kernel=3, pad=1)     → BatchNorm1d → ReLU → MaxPool1d(4)
Conv1d(64→128, kernel=3, pad=1)    → BatchNorm1d → ReLU → MaxPool1d(4)
Conv1d(128→256, kernel=3, pad=1)   → BatchNorm1d → ReLU → AdaptiveAvgPool1d(1)

Dropout(0.5) → Dense(256→128) → ReLU → Dropout(0.3) → Dense(128→10)
```

## Handcrafted Feature Vector

| Feature Group | Features per Group | Stats per Feature | Total |
|--------------|-------------------|-------------------|-------|
| MFCC (40) + delta + delta2 | 120 | 7 (mean, std, min, max, median, skew, kurtosis) | 840 |
| Spectral centroid | 1 | 7 | 7 |
| Spectral bandwidth | 1 | 7 | 7 |
| Spectral rolloff | 1 | 7 | 7 |
| Spectral flatness | 1 | 7 | 7 |
| ZCR | 1 | 7 | 7 |
| RMS | 1 | 7 | 7 |
| Spectral contrast (7 bands) | 7 | 7 | 49 |
| **Total** | | | **~931** |

## Related Docs

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [TECH_STACK.md](TECH_STACK.md)
- [TESTING_PLAN.md](TESTING_PLAN.md)
