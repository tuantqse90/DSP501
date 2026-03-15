"""
DSP501 — Environmental Sound Classification Web Demo

Compare Pipeline A (raw audio) vs Pipeline B (DSP preprocessed) in real time.
Upload any audio file → see classification results side by side.

Run: python app.py
"""

import sys
import os
import pickle

import numpy as np
import librosa
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.models.deep_learning import CNN2D

# ============================================================
# GLOBALS
# ============================================================

MODELS = None
CNN_MODELS = {}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "demo_models.pkl")
CNN_A_PATH = os.path.join(os.path.dirname(__file__), "models", "cnn_a.pt")
CNN_B_PATH = os.path.join(os.path.dirname(__file__), "models", "cnn_b.pt")

CLASS_NAMES = config.CLASS_NAMES
TARGET_SR = config.TARGET_SR
N_SAMPLES = config.N_SAMPLES

# NullShift dark theme colors
BG_COLOR = "#0a0a0a"
FG_COLOR = "#e0e0e0"
ACCENT_A = "#00d4aa"  # Pipeline A — teal
ACCENT_B = "#ff6b6b"  # Pipeline B — coral


# ============================================================
# DSP PIPELINE B (standalone — no config import needed at runtime)
# ============================================================

def dsp_pipeline_b(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Pipeline B: FIR bandpass → pre-emphasis → normalize."""
    # FIR bandpass filter
    nyquist = sr / 2.0
    low = config.FILTER_LOW_FREQ / nyquist
    high = config.FILTER_HIGH_FREQ / nyquist
    coeffs = scipy_signal.firwin(config.FIR_ORDER, [low, high],
                                  pass_zero=False, window="hann")
    y_filt = scipy_signal.filtfilt(coeffs, 1.0, y)

    # Pre-emphasis
    y_emp = np.append(y_filt[0], y_filt[1:] - config.PRE_EMPHASIS_COEFF * y_filt[:-1])

    # Peak normalize
    peak = np.max(np.abs(y_emp))
    if peak > 0:
        y_emp = y_emp / peak

    return y_emp


# ============================================================
# FEATURE EXTRACTION (standalone)
# ============================================================

def extract_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Extract 931-dim handcrafted feature vector (identical to training)."""
    import pandas as pd

    def agg(arr):
        if len(arr) == 0:
            return np.zeros(7)
        s = pd.Series(arr)
        return np.array([
            np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
            np.median(arr), float(s.skew()), float(s.kurtosis()),
        ])

    feats = []

    # MFCC + delta + delta2
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC,
                                  n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                                  fmin=config.FMIN, fmax=config.FMAX)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_all = np.vstack([mfcc, delta, delta2])  # (120, T)
    for i in range(mfcc_all.shape[0]):
        feats.append(agg(mfcc_all[i]))

    # Spectral features
    for feat_fn, kwargs in [
        (librosa.feature.spectral_centroid, {}),
        (librosa.feature.spectral_bandwidth, {}),
        (librosa.feature.spectral_rolloff, {}),
        (librosa.feature.spectral_flatness, {"y": None}),
    ]:
        if "y" in kwargs and kwargs["y"] is None:
            vals = feat_fn(y=y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
        else:
            vals = feat_fn(y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
        feats.append(agg(vals))

    # ZCR, RMS
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=config.N_FFT,
                                               hop_length=config.HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, frame_length=config.N_FFT,
                                hop_length=config.HOP_LENGTH)[0]
    feats.append(agg(zcr))
    feats.append(agg(rms))

    # Spectral contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=config.N_FFT,
                                                   hop_length=config.HOP_LENGTH)
    for i in range(contrast.shape[0]):
        feats.append(agg(contrast[i]))

    return np.concatenate(feats)


def extract_mel(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Extract normalized mel spectrogram for CNN input. Shape: (128, T)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS,
                                        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                                        fmin=config.FMIN, fmax=config.FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)
    mean, std = np.mean(S_db), np.std(S_db)
    if std > 0:
        S_db = (S_db - mean) / std
    return S_db


# ============================================================
# VISUALIZATION
# ============================================================

def make_comparison_figure(y_raw, y_dsp, sr):
    """Create waveform + spectrogram comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), facecolor=BG_COLOR)

    time = np.arange(len(y_raw)) / sr

    for ax in axes.flat:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=FG_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Waveform A
    axes[0, 0].plot(time, y_raw, color=ACCENT_A, linewidth=0.3, alpha=0.8)
    axes[0, 0].set_title("Pipeline A — Raw Waveform", color=ACCENT_A, fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Amplitude", color=FG_COLOR, fontsize=8)

    # Waveform B
    time_b = np.arange(len(y_dsp)) / sr
    axes[0, 1].plot(time_b, y_dsp, color=ACCENT_B, linewidth=0.3, alpha=0.8)
    axes[0, 1].set_title("Pipeline B — DSP Processed", color=ACCENT_B, fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Amplitude", color=FG_COLOR, fontsize=8)

    # Spectrogram A
    S_a = librosa.feature.melspectrogram(y=y_raw, sr=sr, n_mels=config.N_MELS,
                                          n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                                          fmin=config.FMIN, fmax=config.FMAX)
    S_a_db = librosa.power_to_db(S_a, ref=np.max)
    img_a = axes[1, 0].imshow(S_a_db, aspect="auto", origin="lower",
                                cmap="magma", extent=[0, len(y_raw)/sr, 0, sr/2])
    axes[1, 0].set_title("Mel Spectrogram — Raw", color=ACCENT_A, fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Freq (Hz)", color=FG_COLOR, fontsize=8)
    axes[1, 0].set_xlabel("Time (s)", color=FG_COLOR, fontsize=8)

    # Spectrogram B
    S_b = librosa.feature.melspectrogram(y=y_dsp, sr=sr, n_mels=config.N_MELS,
                                          n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                                          fmin=config.FMIN, fmax=config.FMAX)
    S_b_db = librosa.power_to_db(S_b, ref=np.max)
    img_b = axes[1, 1].imshow(S_b_db, aspect="auto", origin="lower",
                                cmap="magma", extent=[0, len(y_dsp)/sr, 0, sr/2])
    axes[1, 1].set_title("Mel Spectrogram — DSP", color=ACCENT_B, fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Freq (Hz)", color=FG_COLOR, fontsize=8)
    axes[1, 1].set_xlabel("Time (s)", color=FG_COLOR, fontsize=8)

    fig.tight_layout(pad=1.5)
    return fig


def make_prediction_figure(probs_a, probs_b):
    """Create side-by-side bar chart of class probabilities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG_COLOR)

    x = np.arange(len(CLASS_NAMES))
    bar_w = 0.6

    for ax, probs, color, title in [
        (ax1, probs_a, ACCENT_A, "Pipeline A (Raw)"),
        (ax2, probs_b, ACCENT_B, "Pipeline B (DSP)"),
    ]:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=FG_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")

        bars = ax.barh(x, probs, height=bar_w, color=color, alpha=0.85, edgecolor="#333")
        ax.set_yticks(x)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8, color=FG_COLOR)
        ax.set_xlabel("Confidence", color=FG_COLOR, fontsize=9)
        ax.set_title(title, color=color, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

        # Highlight top prediction
        top_idx = np.argmax(probs)
        bars[top_idx].set_alpha(1.0)
        bars[top_idx].set_edgecolor("white")
        bars[top_idx].set_linewidth(1.5)
        ax.annotate(f"{probs[top_idx]:.1%}", xy=(probs[top_idx], top_idx),
                     xytext=(5, 0), textcoords="offset points",
                     color="white", fontsize=9, fontweight="bold", va="center")

    fig.tight_layout(pad=1.5)
    return fig


# ============================================================
# INFERENCE
# ============================================================

def load_models():
    """Load trained models (classical ML + CNN)."""
    global MODELS, CNN_MODELS
    if MODELS is not None:
        return MODELS

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run `python train_models.py` first to train and export models."
        )

    with open(MODEL_PATH, "rb") as f:
        MODELS = pickle.load(f)

    # Load CNN weights
    for key, path in [("cnn_a", CNN_A_PATH), ("cnn_b", CNN_B_PATH)]:
        if os.path.exists(path):
            model = CNN2D(n_classes=config.N_CLASSES)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            CNN_MODELS[key] = model
            print(f"  Loaded {key} from {path}")

    return MODELS


def load_audio(file_path: str):
    """Load and preprocess audio file (resample + pad/truncate)."""
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # Pad or truncate to 4 seconds
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:N_SAMPLES]

    return y, TARGET_SR


def classify(audio_path, model_type="Random Forest"):
    """Main classification function for Gradio."""
    if audio_path is None:
        return None, None, "Upload an audio file to classify."

    models = load_models()

    # Load & prep audio
    y_raw, sr = load_audio(audio_path)

    # Pipeline B processing
    y_dsp = dsp_pipeline_b(y_raw, sr)

    # Select model and run inference
    if model_type == "CNN-2D":
        if "cnn_a" not in CNN_MODELS:
            return None, None, "CNN models not found. Run `python train_models.py` first."

        mel_a = extract_mel(y_raw, sr)  # (128, T)
        mel_b = extract_mel(y_dsp, sr)

        with torch.no_grad():
            tensor_a = torch.FloatTensor(mel_a).unsqueeze(0).unsqueeze(0)  # (1,1,128,T)
            tensor_b = torch.FloatTensor(mel_b).unsqueeze(0).unsqueeze(0)
            logits_a = CNN_MODELS["cnn_a"](tensor_a)
            logits_b = CNN_MODELS["cnn_b"](tensor_b)
            probs_a = F.softmax(logits_a, dim=1)[0].numpy()
            probs_b = F.softmax(logits_b, dim=1)[0].numpy()

        feat_info = f"mel spectrogram {mel_a.shape}"
    else:
        # Extract handcrafted features for classical ML
        feat_a = extract_features(y_raw, sr).reshape(1, -1)
        feat_b = extract_features(y_dsp, sr).reshape(1, -1)

        if model_type == "Random Forest":
            model_a_info = models["rf_a"]
            model_b_info = models["rf_b"]
            feat_a_scaled = model_a_info["scaler"].transform(feat_a)
            feat_b_scaled = model_b_info["scaler"].transform(feat_b)
            probs_a = model_a_info["model"].predict_proba(feat_a_scaled)[0]
            probs_b = model_b_info["model"].predict_proba(feat_b_scaled)[0]
        else:  # SVM
            model_a_pipe = models["svm_a"]
            model_b_pipe = models["svm_b"]
            probs_a = model_a_pipe.predict_proba(feat_a)[0]
            probs_b = model_b_pipe.predict_proba(feat_b)[0]

        feat_info = f"{feat_a.shape[1]} features"

    # Predictions
    pred_a = CLASS_NAMES[np.argmax(probs_a)]
    pred_b = CLASS_NAMES[np.argmax(probs_b)]
    conf_a = np.max(probs_a)
    conf_b = np.max(probs_b)

    # Figures
    fig_signal = make_comparison_figure(y_raw, y_dsp, sr)
    fig_probs = make_prediction_figure(probs_a, probs_b)

    # Summary text
    summary = (
        f"## Results — {model_type}\n\n"
        f"| | Pipeline A (Raw) | Pipeline B (DSP) |\n"
        f"|---|---|---|\n"
        f"| **Prediction** | {pred_a} | {pred_b} |\n"
        f"| **Confidence** | {conf_a:.1%} | {conf_b:.1%} |\n"
        f"| **Agree?** | {'Yes' if pred_a == pred_b else 'No'} | |\n\n"
        f"*Audio: {len(y_raw)/sr:.1f}s @ {sr} Hz — {feat_info}*"
    )

    return fig_signal, fig_probs, summary


# ============================================================
# GRADIO UI
# ============================================================

CSS = """
.gradio-container { background-color: #0a0a0a !important; }
.dark { background-color: #0a0a0a !important; }
footer { display: none !important; }
h1, h2, h3, p, label, span { color: #e0e0e0 !important; }
"""

THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="gray",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("JetBrains Mono"),
)


def build_app():
    with gr.Blocks() as app:
        gr.Markdown(
            "# DSP501 — Environmental Sound Classification\n"
            "**Pipeline A** (raw audio) vs **Pipeline B** (DSP preprocessed) — side by side\n\n"
            "Upload any audio → compare how DSP preprocessing affects classification."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                model_select = gr.Radio(
                    choices=["Random Forest", "SVM", "CNN-2D"],
                    value="Random Forest",
                    label="Model",
                )
                classify_btn = gr.Button("Classify", variant="primary", size="lg")

                gr.Markdown(
                    "### UrbanSound8K Classes\n"
                    "air_conditioner, car_horn, children_playing, dog_bark, "
                    "drilling, engine_idling, gun_shot, jackhammer, siren, street_music"
                )

            with gr.Column(scale=3):
                result_text = gr.Markdown(label="Results")
                signal_plot = gr.Plot(label="Signal Comparison")
                prob_plot = gr.Plot(label="Class Probabilities")

        classify_btn.click(
            fn=classify,
            inputs=[audio_input, model_select],
            outputs=[signal_plot, prob_plot, result_text],
        )

        audio_input.change(
            fn=classify,
            inputs=[audio_input, model_select],
            outputs=[signal_plot, prob_plot, result_text],
        )

    return app


if __name__ == "__main__":
    print("Loading models...")
    load_models()
    print("Starting Gradio app...")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False,
               theme=THEME, css=CSS)
