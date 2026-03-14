# Testing Plan — DSP501

> Version: 1.0 | Last updated: 2026-03-13

## Strategy

| Level | Scope | Tool |
|-------|-------|------|
| Unit | Individual functions (filter design, feature extraction) | pytest |
| Integration | Full pipeline (load → process → extract → predict) | pytest |
| Validation | 10-fold CV with UrbanSound8K predefined folds | Custom |

## Unit Tests

### data_loader
- `test_load_metadata()` — CSV loads with expected columns
- `test_load_audio()` — Output shape matches N_SAMPLES
- `test_audio_padding()` — Short clips padded correctly
- `test_fold_split()` — Train/test split respects folds

### dsp_pipeline
- `test_fir_filter_design()` — Coefficients are correct length
- `test_fir_passband()` — Passband frequencies pass through
- `test_fir_stopband()` — Stopband frequencies attenuated
- `test_iir_stability()` — All poles inside unit circle
- `test_pre_emphasis()` — Output matches expected formula
- `test_normalize()` — Output in [-1, 1] range

### feature_extraction
- `test_mfcc_shape()` — MFCC output is (120, T)
- `test_handcrafted_shape()` — Feature vector has consistent length
- `test_mel_spectrogram_shape()` — Output is (N_MELS, T)

### evaluation
- `test_metrics_range()` — All metrics in [0, 1]
- `test_confusion_matrix_shape()` — (N_CLASSES, N_CLASSES)
- `test_paired_t_test()` — Returns valid p-value

## Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific module
pytest tests/test_dsp_pipeline.py -v
```

## Coverage Target

- Overall: >80%
- Critical modules (dsp_pipeline, feature_extraction): >90%

## Related Docs

- [DEV_GUIDE.md](DEV_GUIDE.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
