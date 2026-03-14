# Architecture — DSP501 Environmental Sound Classification

> Version: 1.0 | Last updated: 2026-03-13

## System Design

```mermaid
graph TD
    A[UrbanSound8K Dataset] --> B[Data Loader]
    B --> C{Pipeline Selection}

    C -->|Pipeline A| D[Raw Audio]
    C -->|Pipeline B| E[DSP Preprocessing]

    E --> E1[Bandpass Filter FIR/IIR]
    E1 --> E2[Pre-emphasis]
    E2 --> E3[Amplitude Normalization]
    E3 --> F[Processed Audio]

    D --> G[Feature Extraction]
    F --> G

    G --> G1[Handcrafted Features]
    G --> G2[Mel Spectrogram]

    G1 --> H1[SVM]
    G1 --> H2[Random Forest]
    G2 --> H3[CNN-2D]

    H1 --> I[Evaluation]
    H2 --> I
    H3 --> I

    I --> J[Statistical Comparison]
    J --> K[Report]
```

## Component Diagram

```mermaid
graph LR
    subgraph src
        DL[data_loader.py]
        SA[signal_analysis.py]
        DP[dsp_pipeline.py]
        FE[feature_extraction.py]
        CML[models/classical_ml.py]
        DLM[models/deep_learning.py]
        EV[evaluation.py]
        VIS[visualization.py]
    end

    subgraph config
        CFG[config.py]
    end

    CFG --> DL
    CFG --> SA
    CFG --> DP
    CFG --> FE
    CFG --> CML
    CFG --> DLM
    CFG --> EV
    CFG --> VIS

    DL --> SA
    DL --> DP
    SA --> DP
    DP --> FE
    FE --> CML
    FE --> DLM
    CML --> EV
    DLM --> EV
    EV --> VIS
```

## Data Flow

1. **Loading**: UrbanSound8K CSV metadata → audio file paths → librosa load → resample to 22050 Hz → pad/truncate to 88200 samples
2. **Pipeline A**: Raw audio → extract features directly
3. **Pipeline B**: Raw audio → FIR bandpass (50-10000 Hz) → pre-emphasis (α=0.97) → peak normalize → extract features
4. **Features → ML**: Handcrafted feature vectors → StandardScaler → SVM/RF
5. **Features → DL**: Mel spectrograms (128 bands) → normalize → CNN-2D
6. **Evaluation**: 10-fold CV (predefined folds) → per-fold metrics → aggregate with 95% CI → paired t-test

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| FIR over IIR as primary | Linear phase preserves temporal structure |
| 22050 Hz sample rate | Standard for audio ML; Nyquist covers 0-11025 Hz |
| 50-10000 Hz passband | Removes DC offset + high-freq noise, preserves all class-relevant bands |
| 40 MFCCs | More coefficients capture finer spectral detail for 10-class problem |
| PyTorch over TF | Better MPS support on macOS |

## Related Docs

- [TECH_STACK.md](TECH_STACK.md) — Dependencies and versions
- [MODEL_SPEC.md](MODEL_SPEC.md) — Model architectures
- [ROADMAP.md](ROADMAP.md) — Implementation phases
