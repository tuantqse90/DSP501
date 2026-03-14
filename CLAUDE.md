# CLAUDE.md — DSP501 Environmental Sound Classification

## Project Overview
Environmental Sound Classification system for DSP501 Final Group Project. Compares Pipeline A (raw signal → AI) vs Pipeline B (DSP preprocessing → feature extraction → AI) using the UrbanSound8K dataset.

## Tech Stack
- **Language**: Python 3.10+
- **Audio**: librosa, soundfile, torchaudio
- **DSP**: scipy.signal (FIR/IIR filter design)
- **ML**: scikit-learn (SVM, Random Forest)
- **DL**: PyTorch (CNN-2D, CNN-1D)
- **Visualization**: matplotlib, seaborn
- **Data**: pandas, numpy

## Key Documentation
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Tech Stack: [docs/TECH_STACK.md](docs/TECH_STACK.md)
- Testing: [docs/TESTING_PLAN.md](docs/TESTING_PLAN.md)
- Deployment: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- Security: [docs/SECURITY.md](docs/SECURITY.md)
- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Model Spec: [docs/MODEL_SPEC.md](docs/MODEL_SPEC.md)

## Coding Rules
1. Follow conventions in [docs/DEV_GUIDE.md](docs/DEV_GUIDE.md)
2. Write tests for every new feature (target: >80% coverage)
3. Use type hints throughout
4. All commits follow Conventional Commits format
5. No hardcoded secrets — use .env
6. NEVER shuffle data across UrbanSound8K folds
7. All hyperparameters defined in config.py — no magic numbers in code

## Progress Tracking
**IMPORTANT**: After EVERY completed task, UPDATE [progress.md](progress.md):
- Log date, task, files changed
- Update status (done/in-progress/blocked)
- Record important decisions in Decision Log

## Workflow
1. Read relevant docs BEFORE coding
2. Implement per ROADMAP phases
3. Test → Lint → Commit → Update progress.md
4. Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06

## Dataset Rules
- UrbanSound8K: 10 classes, 8732 clips, 10 predefined folds
- ALWAYS use predefined folds for cross-validation
- Resample all audio to 22050 Hz
- Pad/truncate to 4 seconds (88200 samples)

## NullShift Standards
- Privacy-first: No tracking/analytics
- Terminal aesthetic: Dark mode only
- Solo builder mindset: Pragmatic > Perfect
