"""
Raw signal characterization for UrbanSound8K.

Covers:
- Time-domain analysis (waveform, amplitude stats, ZCR)
- Frequency-domain analysis (FFT, PSD, dominant frequencies)
- Time-frequency analysis (STFT, mel spectrograms, window comparison)
- Wavelet analysis (CWT, DWT, scalogram, multi-resolution)
- Noise analysis (SNR estimation, background noise)
- Signal characterization (stationary vs non-stationary, spectral leakage)
"""

import numpy as np
from scipy import signal as scipy_signal
import librosa
import pywt

import config


# ============================================================
# TIME-DOMAIN ANALYSIS
# ============================================================

def compute_amplitude_stats(y):
    """Compute time-domain amplitude statistics."""
    rms = np.sqrt(np.mean(y ** 2))
    peak = np.max(np.abs(y))
    crest_factor = peak / rms if rms > 0 else 0
    return {
        'mean': float(np.mean(y)),
        'std': float(np.std(y)),
        'rms': float(rms),
        'peak': float(peak),
        'crest_factor': float(crest_factor),
    }


def compute_zcr(y, frame_length=2048, hop_length=512):
    """Compute zero-crossing rate."""
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
    return zcr[0]


# ============================================================
# FREQUENCY-DOMAIN ANALYSIS
# ============================================================

def compute_fft(y, sr=config.TARGET_SR, n_fft=config.N_FFT):
    """Compute FFT magnitude spectrum."""
    Y = np.fft.rfft(y, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    magnitude = np.abs(Y)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    return freqs, magnitude, magnitude_db


def compute_psd(y, sr=config.TARGET_SR, nperseg=config.N_FFT):
    """Compute Power Spectral Density using Welch's method."""
    freqs, psd = scipy_signal.welch(y, fs=sr, nperseg=nperseg)
    return freqs, psd


def find_dominant_frequencies(freqs, magnitude, n_peaks=5):
    """Find top N dominant frequency peaks."""
    peak_indices = np.argsort(magnitude)[-n_peaks:][::-1]
    return [(float(freqs[i]), float(magnitude[i])) for i in peak_indices]


def compute_bandwidth(freqs, psd, threshold=0.5):
    """Compute bandwidth as frequency range containing threshold fraction of total power."""
    total_power = np.sum(psd)
    cumulative = np.cumsum(psd) / total_power
    low_idx = np.searchsorted(cumulative, (1 - threshold) / 2)
    high_idx = np.searchsorted(cumulative, (1 + threshold) / 2)
    return float(freqs[low_idx]), float(freqs[min(high_idx, len(freqs) - 1)])


# ============================================================
# TIME-FREQUENCY ANALYSIS
# ============================================================

def compute_stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH):
    """Compute STFT magnitude spectrogram."""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db


def compute_mel_spectrogram(y, sr=config.TARGET_SR, n_mels=config.N_MELS,
                             n_fft=config.N_FFT, hop_length=config.HOP_LENGTH):
    """Compute mel spectrogram in dB."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                        hop_length=hop_length, fmin=config.FMIN, fmax=config.FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def compare_window_sizes(y, sr=config.TARGET_SR, sizes=(512, 1024, 2048, 4096)):
    """Compute STFT with different window sizes for comparison."""
    results = {}
    for size in sizes:
        S = librosa.stft(y, n_fft=size, hop_length=size // 4, win_length=size)
        results[size] = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return results


# ============================================================
# WAVELET ANALYSIS
# ============================================================

def compute_cwt(y, sr=config.TARGET_SR, wavelet='morl', n_scales=128,
                freq_range=(50, 10000)):
    """
    Compute Continuous Wavelet Transform scalogram.

    Uses Morlet wavelet by default for good time-frequency localization.
    Returns frequencies (Hz) and coefficient magnitudes.
    """
    # Map desired frequency range to CWT scales
    # scale = (center_freq * sr) / freq
    center_freq = pywt.central_frequency(wavelet)
    f_min, f_max = freq_range
    scale_min = center_freq * sr / f_max
    scale_max = center_freq * sr / f_min
    scales = np.geomspace(scale_min, scale_max, num=n_scales)

    # Compute CWT
    coefficients, frequencies = pywt.cwt(y, scales, wavelet, sampling_period=1.0 / sr)

    return frequencies, np.abs(coefficients)


def compute_dwt(y, wavelet='db4', level=None):
    """
    Compute Discrete Wavelet Transform (multi-level decomposition).

    Returns list of (detail_coefficients, approx_coefficients) per level.
    Default wavelet: Daubechies-4 (good for audio).
    """
    if level is None:
        level = pywt.dwt_max_level(len(y), pywt.Wavelet(wavelet).dec_len)
        level = min(level, 8)  # Cap at 8 levels

    coeffs = pywt.wavedec(y, wavelet, level=level)
    # coeffs[0] = approximation, coeffs[1:] = details (coarse → fine)
    return coeffs, level


def compute_dwt_energy(y, sr=config.TARGET_SR, wavelet='db4', level=None):
    """
    Compute energy distribution across DWT frequency bands.

    Returns dict with band info: level, freq_range (Hz), energy, energy_ratio.
    """
    coeffs, n_levels = compute_dwt(y, wavelet, level)

    total_energy = sum(np.sum(c ** 2) for c in coeffs)
    bands = []

    for i, c in enumerate(coeffs):
        energy = float(np.sum(c ** 2))
        if i == 0:
            # Approximation band (lowest frequencies)
            f_high = sr / (2 ** n_levels)
            bands.append({
                'level': f'A{n_levels}',
                'freq_range': (0, f_high),
                'energy': energy,
                'energy_ratio': energy / total_energy if total_energy > 0 else 0,
            })
        else:
            # Detail band at level (n_levels - i + 1)
            detail_level = n_levels - i + 1
            f_low = sr / (2 ** (detail_level))
            f_high = sr / (2 ** (detail_level - 1))
            bands.append({
                'level': f'D{detail_level}',
                'freq_range': (f_low, f_high),
                'energy': energy,
                'energy_ratio': energy / total_energy if total_energy > 0 else 0,
            })

    return bands


def compare_stft_vs_cwt(y, sr=config.TARGET_SR):
    """
    Compute both STFT and CWT for side-by-side comparison.

    Demonstrates time-frequency resolution trade-off:
    - STFT: fixed resolution (uniform time-freq grid)
    - CWT: adaptive resolution (fine time at high freq, fine freq at low freq)
    """
    # STFT
    stft_db = compute_stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)

    # CWT
    cwt_freqs, cwt_coeffs = compute_cwt(y, sr)

    return {
        'stft': stft_db,
        'cwt_freqs': cwt_freqs,
        'cwt_coeffs': cwt_coeffs,
    }


# ============================================================
# NOISE & STATIONARITY ANALYSIS
# ============================================================

def estimate_snr(y, sr=config.TARGET_SR, noise_floor_percentile=10):
    """Estimate SNR by comparing signal power to noise floor estimate."""
    frame_length = int(0.025 * sr)
    hop = int(0.01 * sr)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop)
    frame_power = np.mean(frames ** 2, axis=0)

    noise_threshold = np.percentile(frame_power, noise_floor_percentile)
    signal_power = np.mean(frame_power[frame_power > noise_threshold])
    noise_power = np.mean(frame_power[frame_power <= noise_threshold]) if np.any(frame_power <= noise_threshold) else 1e-10

    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    return float(snr_db)


def compute_spectral_leakage(y, sr=config.TARGET_SR, n_fft=config.N_FFT,
                              windows=('boxcar', 'hann', 'hamming', 'blackman')):
    """Compare spectral leakage across different window functions."""
    results = {}
    for win_name in windows:
        win = scipy_signal.get_window(win_name, len(y) if len(y) <= n_fft else n_fft)
        segment = y[:len(win)]
        windowed = segment * win
        Y = np.fft.rfft(windowed, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        results[win_name] = {
            'freqs': freqs,
            'magnitude_db': 20 * np.log10(np.abs(Y) + 1e-10)
        }
    return results


def check_stationarity(y, sr=config.TARGET_SR, segment_duration=0.5):
    """
    Simple stationarity check by comparing statistics across time segments.

    Returns coefficient of variation of RMS across segments.
    High CV → non-stationary, Low CV → stationary-like.
    """
    segment_samples = int(segment_duration * sr)
    n_segments = len(y) // segment_samples

    if n_segments < 2:
        return {'cv_rms': 0.0, 'stationary': True}

    rms_values = []
    for i in range(n_segments):
        segment = y[i * segment_samples:(i + 1) * segment_samples]
        rms_values.append(np.sqrt(np.mean(segment ** 2)))

    rms_values = np.array(rms_values)
    mean_rms = np.mean(rms_values)
    cv = np.std(rms_values) / mean_rms if mean_rms > 0 else 0

    return {
        'cv_rms': float(cv),
        'rms_per_segment': rms_values.tolist(),
        'stationary': cv < 0.3  # Threshold heuristic
    }
