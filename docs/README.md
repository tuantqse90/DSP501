# DSP501 — Environmental Sound Classification

> // comparing raw signal vs DSP-preprocessed pipelines on UrbanSound8K

## Overview

Environmental Sound Classification (ESC) system that scientifically evaluates whether DSP preprocessing improves AI classification performance. Built for the DSP501 Final Group Project.

**Research Question**: Does applying digital signal processing (bandpass filtering, pre-emphasis, normalization) before feature extraction improve classification accuracy compared to using raw audio?

## Two Pipelines

| Pipeline | Flow | Purpose |
|----------|------|---------|
| **A** (Raw) | Raw Audio → Features → AI Model | Baseline |
| **B** (DSP) | Raw Audio → Filter → Pre-emphasis → Normalize → Features → AI Model | Experimental |

## Dataset

**UrbanSound8K** — 8,732 labeled urban sound clips across 10 classes:
`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`

## Quick Start

```bash
# Clone and setup
cd dsp501-env-sound
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download UrbanSound8K → place in data/UrbanSound8K/

# Run notebooks in order
jupyter notebook notebooks/
```

## Tech Stack

- Python 3.10+ / librosa / scipy / PyTorch / scikit-learn
- See [TECH_STACK.md](TECH_STACK.md) for details

## Project Structure

```
├── config.py                    # Central configuration
├── src/                         # Source modules
│   ├── data_loader.py           # Dataset loading
│   ├── signal_analysis.py       # Signal characterization
│   ├── dsp_pipeline.py          # Filter design & preprocessing
│   ├── feature_extraction.py    # Feature engineering
│   ├── models/                  # SVM, RF, CNN
│   ├── evaluation.py            # Metrics & statistical tests
│   └── visualization.py         # Plotting functions
├── notebooks/                   # 6 ordered notebooks
├── results/                     # Figures & tables
└── report/                      # IEEE format report
```

## Related Docs

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [TECH_STACK.md](TECH_STACK.md)
- [DEV_GUIDE.md](DEV_GUIDE.md)
- [ROADMAP.md](ROADMAP.md)
