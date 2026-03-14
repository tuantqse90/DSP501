"""
Visualization functions for DSP501 project.

All plotting functions save figures to results/figures/ and optionally display them.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

import config

# Global style
plt.style.use('dark_background')
sns.set_palette("bright")

SAVE_DIR = config.FIGURES_DIR


def _savefig(fig, name, dpi=150):
    """Save figure to results/figures/."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, name), dpi=dpi, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')


# ============================================================
# SIGNAL ANALYSIS PLOTS
# ============================================================

def plot_waveforms_per_class(audio_samples, sr=config.TARGET_SR, save=True):
    """Plot one waveform per class in a grid."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.patch.set_facecolor('#0a0a0a')

    for i, (cls_name, y) in enumerate(audio_samples.items()):
        ax = axes[i // 5, i % 5]
        times = np.arange(len(y)) / sr
        ax.plot(times, y, color='#00ff41', linewidth=0.5)
        ax.set_title(cls_name, color='#ffffff', fontsize=10)
        ax.set_xlabel('Time (s)', color='#888888', fontsize=8)
        ax.set_facecolor('#111111')

    plt.suptitle('Waveforms per Class', color='#00ffff', fontsize=14)
    plt.tight_layout()
    if save:
        _savefig(fig, 'fig_waveforms_per_class.png')
    return fig


def plot_fft_per_class(fft_data, save=True):
    """Plot FFT magnitude per class."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.patch.set_facecolor('#0a0a0a')

    for i, (cls_name, (freqs, mag_db)) in enumerate(fft_data.items()):
        ax = axes[i // 5, i % 5]
        ax.plot(freqs, mag_db, color='#00ffff', linewidth=0.5)
        ax.set_title(cls_name, color='#ffffff', fontsize=10)
        ax.set_xlabel('Frequency (Hz)', color='#888888', fontsize=8)
        ax.set_ylabel('dB', color='#888888', fontsize=8)
        ax.set_xlim(0, config.TARGET_SR // 2)
        ax.set_facecolor('#111111')

    plt.suptitle('FFT Magnitude Spectrum per Class', color='#00ffff', fontsize=14)
    plt.tight_layout()
    if save:
        _savefig(fig, 'fig_fft_per_class.png')
    return fig


def plot_spectrogram(S_db, sr=config.TARGET_SR, title='Spectrogram', save_name=None):
    """Plot a single spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')

    img = librosa.display.specshow(S_db, sr=sr, hop_length=config.HOP_LENGTH,
                                    x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title, color='#ffffff')
    plt.tight_layout()

    if save_name:
        _savefig(fig, save_name)
    return fig


def plot_psd_per_class(psd_data, save=True):
    """Plot PSD per class."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.patch.set_facecolor('#0a0a0a')

    for i, (cls_name, (freqs, psd)) in enumerate(psd_data.items()):
        ax = axes[i // 5, i % 5]
        ax.semilogy(freqs, psd, color='#ff0080', linewidth=0.5)
        ax.set_title(cls_name, color='#ffffff', fontsize=10)
        ax.set_xlabel('Frequency (Hz)', color='#888888', fontsize=8)
        ax.set_facecolor('#111111')

    plt.suptitle('Power Spectral Density per Class', color='#00ffff', fontsize=14)
    plt.tight_layout()
    if save:
        _savefig(fig, 'fig_psd_per_class.png')
    return fig


# ============================================================
# DSP PIPELINE PLOTS
# ============================================================

def plot_filter_response(w, mag_db, phase, title='Filter Response', save_name=None):
    """Plot frequency and phase response of a filter."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0a')

    ax1.plot(w, mag_db, color='#00ff41')
    ax1.set_ylabel('Magnitude (dB)', color='#888888')
    ax1.set_title(f'{title} — Magnitude Response', color='#ffffff')
    ax1.axhline(-3, color='#ff0080', linestyle='--', alpha=0.7, label='-3dB')
    ax1.legend()
    ax1.set_facecolor('#111111')
    ax1.grid(True, alpha=0.2)

    ax2.plot(w, phase, color='#00ffff')
    ax2.set_xlabel('Frequency (Hz)', color='#888888')
    ax2.set_ylabel('Phase (radians)', color='#888888')
    ax2.set_title(f'{title} — Phase Response', color='#ffffff')
    ax2.set_facecolor('#111111')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_name:
        _savefig(fig, save_name)
    return fig


def plot_pole_zero(zeros, poles, title='Pole-Zero Plot', save_name=None):
    """Plot pole-zero diagram."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='#888888', linestyle='--', alpha=0.5)

    ax.scatter(np.real(zeros), np.imag(zeros), marker='o', s=80,
               facecolors='none', edgecolors='#00ff41', linewidths=2, label='Zeros')
    ax.scatter(np.real(poles), np.imag(poles), marker='x', s=80,
               color='#ff0080', linewidths=2, label='Poles')

    ax.set_xlabel('Real', color='#888888')
    ax.set_ylabel('Imaginary', color='#888888')
    ax.set_title(title, color='#ffffff')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_name:
        _savefig(fig, save_name)
    return fig


def plot_before_after(y_raw, y_processed, sr=config.TARGET_SR, title='', save_name=None):
    """Plot before/after waveform and spectrum comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor('#0a0a0a')

    times = np.arange(len(y_raw)) / sr

    # Waveforms
    axes[0, 0].plot(times, y_raw, color='#888888', linewidth=0.5, label='Raw')
    axes[0, 0].set_title('Raw Waveform', color='#ffffff')
    axes[0, 0].set_facecolor('#111111')

    times_proc = np.arange(len(y_processed)) / sr
    axes[0, 1].plot(times_proc, y_processed, color='#00ff41', linewidth=0.5, label='Processed')
    axes[0, 1].set_title('Processed Waveform', color='#ffffff')
    axes[0, 1].set_facecolor('#111111')

    # FFT
    from src.signal_analysis import compute_fft
    freqs_r, _, mag_db_r = compute_fft(y_raw, sr)
    freqs_p, _, mag_db_p = compute_fft(y_processed, sr)

    axes[1, 0].plot(freqs_r, mag_db_r, color='#888888', linewidth=0.5)
    axes[1, 0].set_title('Raw FFT', color='#ffffff')
    axes[1, 0].set_xlim(0, sr // 2)
    axes[1, 0].set_facecolor('#111111')

    axes[1, 1].plot(freqs_p, mag_db_p, color='#00ff41', linewidth=0.5)
    axes[1, 1].set_title('Processed FFT', color='#ffffff')
    axes[1, 1].set_xlim(0, sr // 2)
    axes[1, 1].set_facecolor('#111111')

    plt.suptitle(f'Before vs After — {title}', color='#00ffff', fontsize=14)
    plt.tight_layout()
    if save_name:
        _savefig(fig, save_name)
    return fig


# ============================================================
# EVALUATION PLOTS
# ============================================================

def plot_confusion_matrix(cm, title='Confusion Matrix', save_name=None):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0a')

    sns.heatmap(cm, annot=True, fmt='.2f', cmap='magma',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Predicted', color='#ffffff')
    ax.set_ylabel('True', color='#ffffff')
    ax.set_title(title, color='#ffffff')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_name:
        _savefig(fig, save_name)
    return fig


def plot_roc_curves(roc_data, title='ROC Curves', save_name=None):
    """Plot ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')

    colors = ['#00ff41', '#00ffff', '#ff0080', '#a855f7', '#ffaa00',
              '#ff4444', '#44ff44', '#4444ff', '#ff44ff', '#ffff44']

    for i, (cls_name, data) in enumerate(roc_data.items()):
        ax.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)],
                label=f"{cls_name} (AUC={data['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', alpha=0.5)
    ax.set_xlabel('False Positive Rate', color='#ffffff')
    ax.set_ylabel('True Positive Rate', color='#ffffff')
    ax.set_title(title, color='#ffffff')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_name:
        _savefig(fig, save_name)
    return fig


def plot_training_curves(history, title='Training Curves', save_name=None):
    """Plot training/validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0a0a')

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], color='#00ff41', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color='#ff0080', label='Val Loss')
    ax1.set_xlabel('Epoch', color='#888888')
    ax1.set_ylabel('Loss', color='#888888')
    ax1.set_title('Loss', color='#ffffff')
    ax1.legend()
    ax1.set_facecolor('#111111')
    ax1.grid(True, alpha=0.2)

    ax2.plot(epochs, history['train_acc'], color='#00ff41', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], color='#ff0080', label='Val Acc')
    ax2.set_xlabel('Epoch', color='#888888')
    ax2.set_ylabel('Accuracy', color='#888888')
    ax2.set_title('Accuracy', color='#ffffff')
    ax2.legend()
    ax2.set_facecolor('#111111')
    ax2.grid(True, alpha=0.2)

    plt.suptitle(title, color='#00ffff', fontsize=14)
    plt.tight_layout()

    if save_name:
        _savefig(fig, save_name)
    return fig


def plot_accuracy_comparison(results_table, save_name=None):
    """Plot bar chart comparing accuracy across models and pipelines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')

    models = [r['Model'] for r in results_table]
    pipelines = [r['Pipeline'] for r in results_table]
    accuracies = [float(r['Accuracy'].split('±')[0].replace('%', '').strip()) for r in results_table]

    colors = ['#888888' if 'A' in p else '#00ff41' for p in pipelines]
    labels = [f"{m}\n{p}" for m, p in zip(models, pipelines)]

    bars = ax.bar(range(len(labels)), accuracies, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)', color='#ffffff')
    ax.set_title('Model Accuracy Comparison', color='#00ffff')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if save_name:
        _savefig(fig, save_name)
    return fig
