"""
Train and export models for web demo.

Loads feature_cache.pkl, trains RF + SVM + CNN-2D on ALL data (Pipeline A & B),
and saves model files for the Gradio app.

Run once: python train_models.py
"""

import sys
import os
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
from src.models.deep_learning import CNN2D, get_device


def train_cnn_full(mel_data, labels, tag="CNN", epochs=60, batch_size=32, lr=0.001):
    """Train CNN-2D on full dataset with 90/10 split for early stopping."""
    device = get_device()
    print(f"  Device: {device}")

    # 90/10 split (use fold 10 as val to respect UrbanSound8K structure)
    n = len(labels)
    idx = np.arange(n)
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(idx)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.FloatTensor(mel_data[train_idx]).unsqueeze(1)  # (N, 1, 128, T)
    y_train = torch.LongTensor(labels[train_idx].astype(int))
    X_val = torch.FloatTensor(mel_data[val_idx]).unsqueeze(1)
    y_val = torch.LongTensor(labels[val_idx].astype(int))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                             batch_size=batch_size, shuffle=False)

    model = CNN2D(n_classes=config.N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (out.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        t_loss = train_loss / train_total
        v_loss = val_loss / val_total
        t_acc = train_correct / train_total
        v_acc = val_correct / val_total
        scheduler.step(v_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} — "
                  f"train_loss={t_loss:.4f} train_acc={t_acc:.3f} | "
                  f"val_loss={v_loss:.4f} val_acc={v_acc:.3f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.cpu().eval()
    print(f"  Best val_loss={best_val_loss:.4f}, val_acc={v_acc:.3f}")
    return model


def main():
    print("Loading feature cache...")
    cache_path = os.path.join(config.RESULTS_DIR, "feature_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    X_raw = cache["features_raw"]    # (8732, 931)
    X_dsp = cache["features_dsp"]    # (8732, 931)
    mel_raw = cache["mel_raw"]       # (8732, 128, 173)
    mel_dsp = cache["mel_dsp"]       # (8732, 128, 173)
    y = cache["labels"]              # (8732,)

    print(f"Features loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} dims")
    print(f"Mel spectrograms: {mel_raw.shape}")
    print(f"Classes: {config.CLASS_NAMES}")

    models = {}

    # --- Random Forest ---
    print("\n[1/6] Training RF Pipeline A...")
    scaler_rf_a = StandardScaler()
    X_rf_a = scaler_rf_a.fit_transform(X_raw)
    rf_a = RandomForestClassifier(n_estimators=500, max_depth=None,
                                   random_state=config.RANDOM_SEED, n_jobs=-1)
    rf_a.fit(X_rf_a, y)
    models["rf_a"] = {"model": rf_a, "scaler": scaler_rf_a}
    print(f"  Train accuracy: {rf_a.score(X_rf_a, y):.4f}")

    print("[2/6] Training RF Pipeline B...")
    scaler_rf_b = StandardScaler()
    X_rf_b = scaler_rf_b.fit_transform(X_dsp)
    rf_b = RandomForestClassifier(n_estimators=500, max_depth=None,
                                   random_state=config.RANDOM_SEED, n_jobs=-1)
    rf_b.fit(X_rf_b, y)
    models["rf_b"] = {"model": rf_b, "scaler": scaler_rf_b}
    print(f"  Train accuracy: {rf_b.score(X_rf_b, y):.4f}")

    # --- SVM ---
    print("[3/6] Training SVM Pipeline A...")
    svm_pipe_a = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=200, random_state=config.RANDOM_SEED)),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                     probability=True, random_state=config.RANDOM_SEED)),
    ])
    svm_pipe_a.fit(X_raw, y)
    models["svm_a"] = svm_pipe_a
    print(f"  Train accuracy: {svm_pipe_a.score(X_raw, y):.4f}")

    print("[4/6] Training SVM Pipeline B...")
    svm_pipe_b = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=200, random_state=config.RANDOM_SEED)),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                     probability=True, random_state=config.RANDOM_SEED)),
    ])
    svm_pipe_b.fit(X_dsp, y)
    models["svm_b"] = svm_pipe_b
    print(f"  Train accuracy: {svm_pipe_b.score(X_dsp, y):.4f}")

    # --- CNN-2D ---
    print("[5/6] Training CNN-2D Pipeline A...")
    cnn_a = train_cnn_full(mel_raw, y, tag="CNN-A")
    torch.save(cnn_a.state_dict(),
               os.path.join(os.path.dirname(__file__), "models", "cnn_a.pt"))

    print("[6/6] Training CNN-2D Pipeline B...")
    cnn_b = train_cnn_full(mel_dsp, y, tag="CNN-B")
    torch.save(cnn_b.state_dict(),
               os.path.join(os.path.dirname(__file__), "models", "cnn_b.pt"))

    # Save classical models (RF + SVM)
    out_path = os.path.join(os.path.dirname(__file__), "models", "demo_models.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nClassical models saved: {out_path} ({size_mb:.1f} MB)")
    print("CNN weights saved: models/cnn_a.pt, models/cnn_b.pt")
    print("Done!")


if __name__ == "__main__":
    main()
