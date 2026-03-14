"""
Central configuration for DSP501 Environmental Sound Classification project.
All hyperparameters, paths, and random seeds defined here for reproducibility.
"""

import os

# === Reproducibility ===
RANDOM_SEED = 42
N_FOLDS = 10  # UrbanSound8K predefined folds

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "UrbanSound8K")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
METADATA_PATH = os.path.join(DATA_DIR, "metadata", "UrbanSound8K.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# === Audio Parameters ===
TARGET_SR = 22050          # Resample all audio to this rate
AUDIO_DURATION = 4.0       # Max duration in seconds
N_SAMPLES = int(TARGET_SR * AUDIO_DURATION)  # 88,200 samples

# === DSP Parameters ===
FILTER_TYPE = "bandpass"
FILTER_LOW_FREQ = 50       # Hz — remove DC offset and very low noise
FILTER_HIGH_FREQ = 10000   # Hz — remove high-freq noise above useful range
FIR_ORDER = 101            # FIR filter order (odd number)
IIR_ORDER = 5              # Butterworth IIR order
PRE_EMPHASIS_COEFF = 0.97

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
N_CLASSES = len(CLASS_NAMES)
