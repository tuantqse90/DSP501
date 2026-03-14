"""
Feature extraction for environmental sound classification.

Extracts:
- MFCC + delta + delta-delta
- Spectral features (centroid, bandwidth, rolloff, contrast, flatness)
- Temporal features (ZCR, RMS, onset strength)
- Statistical aggregation over time frames
- Mel spectrogram for CNN input
"""

import numpy as np
import librosa
import pandas as pd

import config


# ============================================================
# SPECTRAL FEATURES
# ============================================================

def extract_mfcc(y, sr=config.TARGET_SR):
    """Extract MFCC + delta + delta-delta features."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC,
                                 n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                                 fmin=config.FMIN, fmax=config.FMAX)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])  # (3*N_MFCC, T)


def extract_spectral_features(y, sr=config.TARGET_SR):
    """Extract spectral features frame-by-frame."""
    features = {}
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
    features['spectral_contrast'] = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    features['spectral_flatness'] = librosa.feature.spectral_flatness(
        y=y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
    features['zcr'] = librosa.feature.zero_crossing_rate(
        y=y, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
    features['rms'] = librosa.feature.rms(
        y=y, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
    return features


# ============================================================
# STATISTICAL AGGREGATION
# ============================================================

def aggregate_stats(feature_array):
    """Compute statistics over time axis for a 1D feature sequence."""
    if len(feature_array) == 0:
        return np.zeros(7)
    return np.array([
        np.mean(feature_array),
        np.std(feature_array),
        np.min(feature_array),
        np.max(feature_array),
        np.median(feature_array),
        float(pd.Series(feature_array).skew()),
        float(pd.Series(feature_array).kurtosis()),
    ])


# ============================================================
# COMBINED FEATURE EXTRACTION
# ============================================================

def extract_handcrafted_features(y, sr=config.TARGET_SR):
    """
    Extract a fixed-length feature vector from an audio clip.

    Combines MFCC stats + spectral feature stats.

    Returns:
        np.ndarray of shape (n_features,)
    """
    all_features = []

    # MFCC + delta + delta2 — stats per coefficient
    mfcc_all = extract_mfcc(y, sr)  # (120, T)
    for i in range(mfcc_all.shape[0]):
        all_features.append(aggregate_stats(mfcc_all[i]))

    # Spectral features — stats per feature
    spectral = extract_spectral_features(y, sr)
    for key in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                'spectral_flatness', 'zcr', 'rms']:
        all_features.append(aggregate_stats(spectral[key]))

    # Spectral contrast — stats per band
    for i in range(spectral['spectral_contrast'].shape[0]):
        all_features.append(aggregate_stats(spectral['spectral_contrast'][i]))

    return np.concatenate(all_features)


def extract_mel_spectrogram(y, sr=config.TARGET_SR):
    """
    Extract mel spectrogram for CNN input.

    Returns:
        np.ndarray of shape (N_MELS, T) in dB scale, normalized.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS,
                                        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                                        fmin=config.FMIN, fmax=config.FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize per sample
    mean = np.mean(S_db)
    std = np.std(S_db)
    if std > 0:
        S_db = (S_db - mean) / std

    return S_db


def extract_all_features(audio_list, sr=config.TARGET_SR, show_progress=True):
    """
    Extract handcrafted features for a list of audio signals.

    Args:
        audio_list: List of np.ndarray audio signals.
        sr: Sampling rate.

    Returns:
        np.ndarray of shape (n_samples, n_features)
    """
    from tqdm import tqdm

    features = []
    iterator = tqdm(audio_list, desc="Extracting features") if show_progress else audio_list
    for y in iterator:
        features.append(extract_handcrafted_features(y, sr))

    return np.array(features)
