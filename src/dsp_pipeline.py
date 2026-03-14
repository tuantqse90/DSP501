"""
Digital filter design and DSP preprocessing pipeline (Pipeline B).

Implements:
- FIR bandpass filter (window method)
- IIR Butterworth bandpass filter
- Pre-emphasis filter
- Amplitude normalization
- Silence removal
- Before/after comparison utilities
"""

import numpy as np
from scipy import signal as scipy_signal

import config


# ============================================================
# FIR FILTER
# ============================================================

def design_fir_bandpass(low_freq=config.FILTER_LOW_FREQ, high_freq=config.FILTER_HIGH_FREQ,
                         sr=config.TARGET_SR, order=config.FIR_ORDER):
    """
    Design FIR bandpass filter using window method (Hann).

    Returns:
        coefficients: np.ndarray of FIR filter taps.
    """
    nyquist = sr / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    coeffs = scipy_signal.firwin(order, [low, high], pass_zero=False, window='hann')
    return coeffs


def apply_fir_filter(y, coeffs):
    """Apply FIR filter using zero-phase filtering (filtfilt)."""
    return scipy_signal.filtfilt(coeffs, 1.0, y)


def fir_frequency_response(coeffs, sr=config.TARGET_SR, n_points=8192):
    """Compute frequency response of FIR filter."""
    w, h = scipy_signal.freqz(coeffs, worN=n_points, fs=sr)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
    phase = np.unwrap(np.angle(h))
    return w, magnitude_db, phase


def fir_group_delay(coeffs, sr=config.TARGET_SR, n_points=8192):
    """Compute group delay of FIR filter."""
    w, gd = scipy_signal.group_delay((coeffs, 1.0), w=n_points, fs=sr)
    return w, gd


def fir_impulse_response(coeffs):
    """Return impulse response (the coefficients themselves for FIR)."""
    return coeffs


# ============================================================
# IIR BUTTERWORTH FILTER
# ============================================================

def design_iir_bandpass(low_freq=config.FILTER_LOW_FREQ, high_freq=config.FILTER_HIGH_FREQ,
                         sr=config.TARGET_SR, order=config.IIR_ORDER):
    """
    Design IIR Butterworth bandpass filter.

    Returns:
        (b, a): Numerator and denominator coefficients.
    """
    nyquist = sr / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return b, a


def apply_iir_filter(y, b, a):
    """Apply IIR filter using zero-phase filtering (filtfilt)."""
    return scipy_signal.filtfilt(b, a, y)


def iir_frequency_response(b, a, sr=config.TARGET_SR, n_points=8192):
    """Compute frequency response of IIR filter."""
    w, h = scipy_signal.freqz(b, a, worN=n_points, fs=sr)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
    phase = np.unwrap(np.angle(h))
    return w, magnitude_db, phase


def iir_pole_zero(b, a):
    """Compute poles and zeros of IIR filter."""
    zeros = np.roots(b)
    poles = np.roots(a)
    return zeros, poles


def check_stability(a):
    """Check if IIR filter is stable (all poles inside unit circle)."""
    poles = np.roots(a)
    return bool(np.all(np.abs(poles) < 1.0))


# ============================================================
# ADDITIONAL PREPROCESSING
# ============================================================

def pre_emphasis(y, coeff=config.PRE_EMPHASIS_COEFF):
    """Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]."""
    return np.append(y[0], y[1:] - coeff * y[:-1])


def normalize_amplitude(y):
    """Peak normalization to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak > 0:
        return y / peak
    return y


def remove_silence(y, sr=config.TARGET_SR, top_db=20):
    """Trim leading and trailing silence."""
    import librosa
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


# ============================================================
# FULL PIPELINE
# ============================================================

def pipeline_b_process(y, sr=config.TARGET_SR, use_fir=True):
    """
    Full Pipeline B preprocessing chain:
    1. Bandpass filter (FIR or IIR)
    2. Pre-emphasis
    3. Amplitude normalization

    Args:
        y: Raw audio signal.
        sr: Sampling rate.
        use_fir: If True, use FIR filter; else use IIR.

    Returns:
        Processed signal.
    """
    # Bandpass filter
    if use_fir:
        coeffs = design_fir_bandpass(sr=sr)
        y_filtered = apply_fir_filter(y, coeffs)
    else:
        b, a = design_iir_bandpass(sr=sr)
        y_filtered = apply_iir_filter(y, b, a)

    # Pre-emphasis
    y_emphasized = pre_emphasis(y_filtered)

    # Normalize
    y_normalized = normalize_amplitude(y_emphasized)

    return y_normalized


def compare_before_after(y_raw, y_processed, sr=config.TARGET_SR):
    """
    Compute comparison metrics between raw and processed signals.

    Returns dict with SNR improvement, spectral differences, etc.
    """
    from src.signal_analysis import estimate_snr, compute_psd

    snr_raw = estimate_snr(y_raw, sr)
    snr_processed = estimate_snr(y_processed, sr)

    return {
        'snr_raw': snr_raw,
        'snr_processed': snr_processed,
        'snr_improvement': snr_processed - snr_raw,
        'rms_raw': float(np.sqrt(np.mean(y_raw ** 2))),
        'rms_processed': float(np.sqrt(np.mean(y_processed ** 2))),
    }
