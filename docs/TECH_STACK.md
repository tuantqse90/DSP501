# Tech Stack — DSP501

> Version: 1.0 | Last updated: 2026-03-13

## Core Dependencies

| Package | Version | Purpose | Why Chosen |
|---------|---------|---------|------------|
| numpy | >=1.24.0 | Array operations | Industry standard |
| scipy | >=1.10.0 | DSP (filter design, signal processing) | Full filter design toolkit |
| librosa | >=0.10.0 | Audio loading, feature extraction | Best Python audio analysis library |
| soundfile | >=0.12.0 | Audio I/O backend for librosa | Required by librosa |
| matplotlib | >=3.7.0 | Plotting and visualization | Most flexible Python plotting |
| seaborn | >=0.12.0 | Statistical visualization | Better heatmaps, box plots |
| pandas | >=2.0.0 | Data manipulation, CSV loading | Standard for tabular data |
| scikit-learn | >=1.3.0 | SVM, RF, preprocessing, metrics | Complete ML toolkit |
| torch | >=2.0.0 | CNN models, GPU/MPS training | Best DL framework for research |
| torchaudio | >=2.0.0 | Audio transforms (optional) | PyTorch audio ecosystem |
| tqdm | >=4.65.0 | Progress bars | Better UX during long operations |
| jupyter | >=1.0.0 | Interactive notebooks | Standard for exploratory analysis |

## Alternatives Considered

| Chosen | Alternative | Reason for Choice |
|--------|-------------|-------------------|
| librosa | torchaudio | librosa has richer feature extraction API |
| PyTorch | TensorFlow | Better MPS support, cleaner research API |
| scipy.signal | Custom DSP | scipy is well-tested, complete filter design |
| Jupyter | Scripts only | Notebooks better for exploration and visualization |

## System Requirements

- Python 3.10+
- macOS / Linux / Windows
- 8GB+ RAM recommended (dataset fits in memory)
- GPU optional (MPS on Mac, CUDA on Linux/Windows)

## Related Docs

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEV_GUIDE.md](DEV_GUIDE.md)
