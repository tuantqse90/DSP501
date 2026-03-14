"""
Dataset loading and fold management for UrbanSound8K.

Handles:
- Metadata CSV loading
- Audio file loading with resampling
- Pad/truncate to fixed length
- Fold-based train/test splits (respects predefined folds)
- Class distribution analysis
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import config


def load_metadata():
    """Load UrbanSound8K metadata CSV."""
    df = pd.read_csv(config.METADATA_PATH)
    return df


def load_audio(file_path, sr=config.TARGET_SR, duration=config.AUDIO_DURATION):
    """
    Load a single audio file, resample, and pad/truncate to fixed length.

    Args:
        file_path: Path to audio file.
        sr: Target sampling rate.
        duration: Target duration in seconds.

    Returns:
        np.ndarray of shape (N_SAMPLES,)
    """
    n_samples = int(sr * duration)
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(n_samples)

    # Pad or truncate
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)), mode='constant')
    else:
        y = y[:n_samples]

    return y


def get_file_path(row):
    """Get full file path from metadata row."""
    return os.path.join(config.AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])


def get_fold_split(metadata, test_fold):
    """
    Split metadata into train and test based on UrbanSound8K predefined folds.

    Args:
        metadata: DataFrame from load_metadata().
        test_fold: Fold number (1-10) to use as test set.

    Returns:
        (train_df, test_df)
    """
    test_df = metadata[metadata['fold'] == test_fold]
    train_df = metadata[metadata['fold'] != test_fold]
    return train_df, test_df


def load_fold_data(metadata, fold_ids, sr=config.TARGET_SR, show_progress=True):
    """
    Load all audio files for given fold IDs.

    Args:
        metadata: DataFrame from load_metadata().
        fold_ids: List of fold numbers to load.
        sr: Target sampling rate.
        show_progress: Show tqdm progress bar.

    Returns:
        X: np.ndarray of shape (n_samples, N_SAMPLES)
        y: np.ndarray of shape (n_samples,) — class IDs
    """
    subset = metadata[metadata['fold'].isin(fold_ids)]
    X, y = [], []

    iterator = tqdm(subset.iterrows(), total=len(subset), desc="Loading audio") if show_progress else subset.iterrows()

    for _, row in iterator:
        file_path = get_file_path(row)
        if os.path.exists(file_path):
            audio = load_audio(file_path, sr=sr)
            X.append(audio)
            y.append(row['classID'])

    return np.array(X), np.array(y)


def analyze_class_distribution(metadata):
    """
    Analyze and return class distribution statistics.

    Args:
        metadata: DataFrame from load_metadata().

    Returns:
        dict with class counts and percentages.
    """
    counts = metadata['class'].value_counts()
    total = len(metadata)
    distribution = {
        cls: {'count': int(counts[cls]), 'percentage': round(counts[cls] / total * 100, 2)}
        for cls in config.CLASS_NAMES if cls in counts.index
    }
    return distribution
