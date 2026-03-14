# Deployment — DSP501

> Version: 1.0 | Last updated: 2026-03-13

## Overview

This is a research/academic project — deployment focuses on reproducibility rather than production serving.

## Environment Setup

```bash
# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify GPU/MPS availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

## Dataset

UrbanSound8K must be downloaded separately (CC BY-NC 3.0 license):
1. Request access at https://urbansounddataset.weebly.com/urbansound8k.html
2. Extract to `data/UrbanSound8K/`
3. Verify: `ls data/UrbanSound8K/audio/fold1/` should list .wav files

## Reproducibility

- All random seeds set in `config.py` (RANDOM_SEED=42)
- UrbanSound8K predefined folds ensure consistent splits
- `requirements.txt` pins minimum versions
- Results saved as pickle files in `results/`

## CI/CD

Not applicable for academic project. All experiments run locally via Jupyter notebooks.

## Related Docs

- [DEV_GUIDE.md](DEV_GUIDE.md)
- [TECH_STACK.md](TECH_STACK.md)
