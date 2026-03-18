# Sơ đồ luồng dữ liệu & mã nguồn — DSP501

> Version: 2.0 | Last updated: 2026-03-18

---

## 1. Tổng quan kiến trúc
Sơ đồ ASCII art (hiển thị mọi nơi, không cần Mermaid renderer):
  1. Tổng quan kiến trúc — Full pipeline từ .wav → evaluation
  2. Data Shape flow — Shape biến đổi qua từng bước
  3. 10-Fold CV — Grid hiển thị train/test splits
  4. Pipeline A vs B — Song song, chi tiết 3 bước DSP
  5. Feature Extraction 931-dim — 3 nhánh MFCC/Spectral/Contrast
  6. CNN-2D — 4 conv blocks + classifier
  7. Evaluation — t-test/Wilcoxon/Cohen's d → decision
  8. Source code dependency — Module tree
  9. Phân tích → Quyết định → Code — 3-column tracing

```
  UrbanSound8K (.wav)
  8732 clips × 10 classes × 10 folds
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │               data_loader.py                        │
  │  load_metadata() → load_audio()                     │
  │  • Resample → 22050 Hz                              │
  │  • Pad/Truncate → 88200 samples (4 giây)            │
  │  • get_fold_split(test_fold=i)                      │
  └────────────────────────┬────────────────────────────┘
                           │
                    X: (n_samples, 88200)
                    y: (n_samples,)
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
  ┌───────────────────┐    ┌──────────────────────────────┐
  │   Pipeline A      │    │   Pipeline B                  │
  │   (Raw — Baseline)│    │   (DSP Preprocessing)         │
  │                   │    │   dsp_pipeline.py              │
  │   Không xử lý     │    │                               │
  │   Dùng trực tiếp  │    │  ① FIR Bandpass 50–10000 Hz   │
  │   x[n] gốc        │    │  ② Pre-emphasis (α = 0.97)    │
  │                   │    │  ③ Peak Normalize → [−1, 1]   │
  └────────┬──────────┘    └──────────────┬───────────────┘
           │                              │
           ▼                              ▼
  ┌────────────────────────────────────────────────────────┐
  │              feature_extraction.py                     │
  ├────────────────────────┬───────────────────────────────┤
  │   ML path              │   DL path                     │
  │                        │                               │
  │   extract_handcrafted  │   extract_mel_spectrogram()   │
  │   _features()          │                               │
  │                        │   Mel Spectrogram              │
  │   931-dim vector       │   (128 × 345)                 │
  │   ┌──────────────────┐ │   ┌─────────────────────────┐ │
  │   │ MFCC+Δ+Δ²:  840 │ │   │ 128 mel bands           │ │
  │   │ Spectral:     42 │ │   │ × ~345 time frames      │ │
  │   │ Contrast:     49 │ │   │ dB scale, normalized    │ │
  │   │ ──────────────── │ │   │                         │ │
  │   │ TOTAL:       931 │ │   │ Input: (B, 1, 128, 345) │ │
  │   └──────────────────┘ │   └─────────────────────────┘ │
  └───────────┬────────────┴──────────────┬────────────────┘
              │                           │
              ▼                           ▼
  ┌─────────────────────┐    ┌─────────────────────────────┐
  │  Classical ML       │    │  Deep Learning               │
  │  models/            │    │  models/                     │
  │  classical_ml.py    │    │  deep_learning.py            │
  │                     │    │                              │
  │  ┌───────────────┐  │    │  ┌────────────────────────┐  │
  │  │ SVM (RBF)     │  │    │  │ CNN-2D (4-layer)       │  │
  │  │ GridSearchCV  │  │    │  │ Conv→BN→ReLU→Pool ×4   │  │
  │  │ C, γ tuning   │  │    │  │ FC(256→128→10)         │  │
  │  ├───────────────┤  │    │  │ Adam, lr=0.001         │  │
  │  │ Random Forest │  │    │  │ EarlyStopping(10)      │  │
  │  │ GridSearchCV  │  │    │  │ ReduceLROnPlateau      │  │
  │  │ n_est, depth  │  │    │  └────────────────────────┘  │
  │  └───────────────┘  │    │                              │
  └──────────┬──────────┘    └──────────────┬───────────────┘
             │                              │
             ▼                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                  evaluation.py                         │
  │                                                        │
  │  ┌──────────────────┐  ┌─────────────────────────────┐ │
  │  │ Per-fold Metrics  │  │ Statistical Comparison      │ │
  │  │ • Accuracy        │  │ • Paired t-test (p-value)   │ │
  │  │ • Precision       │  │ • Wilcoxon signed-rank      │ │
  │  │ • Recall          │  │ • Cohen's d (effect size)   │ │
  │  │ • F1 (macro)      │  │ • 95% CI                    │ │
  │  └──────────────────┘  └─────────────────────────────┘ │
  │                                                        │
  │   Kết luận: p > 0.05 → DSP không cải thiện accuracy    │
  └────────────────────────────────────────────────────────┘
```

---

## 2. Luồng Data Shape qua từng bước

```
  ┌──────────────┐    librosa.load()     ┌──────────────┐   pipeline_b()    ┌──────────────┐
  │  .wav file   │───── resample  ──────▶│ Loaded audio │───── FIR+PE+N ──▶│  After DSP   │
  │  (N,)        │      + pad            │ (88200,)     │                   │  (88200,)    │
  │  sr khác nhau│                       │ 4s × 22050Hz │                   │  filtered    │
  └──────────────┘                       └──────┬───────┘                   └──────┬───────┘
                                                │                                  │
                          ┌─────────────────────┼──────────────────────────────────┘
                          │                     │
                          ▼                     ▼
                 ┌─────────────────┐   ┌─────────────────────┐
                 │   extract_mfcc  │   │ extract_mel_spectro  │
                 │   (120, ~173)   │   │   gram()             │
                 │                 │   │   (128, ~345)        │
                 └────────┬────────┘   └──────────┬───────────┘
                          │                       │
                          ▼                       ▼
                 ┌─────────────────┐   ┌─────────────────────┐
                 │ aggregate_stats │   │   Normalize + add   │
                 │ + spectral(42)  │   │   channel dim       │
                 │ + contrast(49)  │   │                     │
                 │                 │   │   (1, 128, ~345)    │
                 │    (931,)       │   │   CNN input         │
                 └────────┬────────┘   └──────────┬───────────┘
                          │                       │
                          ▼                       ▼
                 ┌─────────────────┐   ┌─────────────────────┐
                 │  SVM / RF       │   │    CNN-2D            │
                 │  (n, 931)→(n,10)│   │ (B,1,128,345)→(B,10)│
                 └────────┬────────┘   └──────────┬───────────┘
                          │                       │
                          └───────────┬───────────┘
                                      ▼
                             ┌─────────────────┐
                             │  Predictions     │
                             │  (n_test, 10)    │
                             │  → argmax → (n,) │
                             └─────────────────┘
```

### Bảng Data Shape chi tiết

| Bước | Shape | Kích thước | Giải thích |
|------|-------|------------|------------|
| Raw `.wav` | $(N,)$ | Khác nhau | File gốc, $f_s$ khác nhau |
| Loaded audio | $(88200,)$ | 88200 float32 | $4\text{s} \times 22050\text{Hz}$, mono |
| Batch (1 fold) | $(\sim\!873, 88200)$ | ~308 MB | ~873 samples/fold |
| After DSP | $(88200,)$ | 88200 float32 | Filtered + normalized |
| MFCC + $\Delta$ + $\Delta^2$ | $(120, \sim\!173)$ | 20760 values | $40 \times 3$ coeff $\times$ $T$ frames |
| Spectral features | `dict{7 keys}` | $7 \times \sim\!173$ | Per-frame features |
| Handcrafted vector | $(931,)$ | 931 float64 | Fixed-length cho ML |
| Feature matrix | $(\sim\!873, 931)$ | ~3.2 MB | 1 fold ML input |
| Mel spectrogram | $(128, \sim\!345)$ | 44160 values | 128 mel $\times$ $T$ frames |
| CNN batch | $(32, 1, 128, \sim\!345)$ | ~5.6 MB | Batch size 32 |
| CNN output | $(32, 10)$ | 320 values | 10 class logits |
| Fold predictions | $(\sim\!873,)$ | 873 int | Class IDs $0$–$9$ |
| 10-fold accuracy | $(10,)$ | 10 float | Accuracy mỗi fold |

---

## 3. Luồng 10-Fold Cross-Validation

```
  UrbanSound8K: 10 predefined folds (KHÔNG shuffle)
  ═══════════════════════════════════════════════════

  Iter 1:  [TEST]  Train   Train   Train   Train   Train   Train   Train   Train   Train
            Fold1   Fold2   Fold3   Fold4   Fold5   Fold6   Fold7   Fold8   Fold9   Fold10

  Iter 2:  Train  [TEST]  Train   Train   Train   Train   Train   Train   Train   Train
            Fold1   Fold2   Fold3   Fold4   Fold5   Fold6   Fold7   Fold8   Fold9   Fold10

  Iter 3:  Train   Train  [TEST]  Train   Train   Train   Train   Train   Train   Train
                  ...

  Iter 10: Train   Train   Train   Train   Train   Train   Train   Train   Train  [TEST]
            Fold1   Fold2   Fold3   Fold4   Fold5   Fold6   Fold7   Fold8   Fold9   Fold10

                                         │
                                         ▼
                              acc₁, acc₂, ..., acc₁₀
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │         Aggregate Results              │
                    │                    ___                 │
                    │   Mean:  x̄ = (1/N) Σ  accᵢ            │
                    │                                       │
                    │   95% CI:  x̄ ± 1.96 · s / √N         │
                    │                                       │
                    │   N = 10 folds                        │
                    └────────────────────────────────────────┘
```

**Tại sao dùng predefined folds?**
- UrbanSound8K được thiết kế để **các clip từ cùng nguồn gốc nằm trong cùng fold**
- Nếu shuffle ngẫu nhiên → data leakage (clip cùng recording nằm cả train lẫn test)
- **KHÔNG BAO GIỜ** shuffle data across folds

---

## 4. Pipeline A vs Pipeline B — So sánh chi tiết

```
                        ┌──────────────────┐
                        │   Raw Audio x[n]  │
                        │   (88200,)        │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
  ╔═══════════════════════════╗  ╔════════════════════════════════════════╗
  ║   PIPELINE A — Raw        ║  ║   PIPELINE B — DSP                    ║
  ║   (Baseline)              ║  ║   dsp_pipeline.py                     ║
  ║                           ║  ║                                       ║
  ║   Không xử lý gì          ║  ║   ① FIR Bandpass Filter               ║
  ║                           ║  ║      design_fir_bandpass()             ║
  ║   x[n] → Features        ║  ║      50–10000 Hz, order=101            ║
  ║                           ║  ║      Hann window, zero-phase           ║
  ║                           ║  ║                    │                   ║
  ║                           ║  ║                    ▼                   ║
  ║                           ║  ║   ② Pre-emphasis Filter                ║
  ║                           ║  ║      y[n] = x[n] − 0.97·x[n−1]       ║
  ║                           ║  ║      H(z) = 1 − αz⁻¹                  ║
  ║                           ║  ║                    │                   ║
  ║                           ║  ║                    ▼                   ║
  ║                           ║  ║   ③ Peak Normalize                     ║
  ║                           ║  ║      ŷ[n] = y[n] / max|y[n]|          ║
  ║                           ║  ║      Output: [−1, 1]                  ║
  ║                           ║  ║                                       ║
  ╚═════════════╤═════════════╝  ╚════════════════════╤═══════════════════╝
                │                                     │
                └──────────────┬──────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Extraction   │
                    │ (giống nhau cho cả  │
                    │  Pipeline A và B)    │
                    └─────────────────────┘
```

### Công thức DSP trong Pipeline B

**FIR Bandpass Filter** — phương pháp cửa sổ:

$$y[n] = \sum_{k=0}^{M} h[k] \cdot x[n-k], \quad M = 100 \text{ (order 101)}$$

Đáp ứng tần số lý tưởng:

$$|H(f)| = \begin{cases} 1 & \text{nếu } f_L \leq |f| \leq f_H \\ 0 & \text{ngược lại} \end{cases}, \quad f_L = 50\text{ Hz}, \quad f_H = 10000\text{ Hz}$$

**Pre-emphasis** — bộ lọc thông cao bậc 1:

$$y[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha = 0.97$$

$$H(z) = 1 - \alpha z^{-1}$$

**Peak Normalization:**

$$\hat{y}[n] = \frac{y[n]}{\displaystyle\max_{n} |y[n]|}$$

### So sánh FIR vs IIR

```
  ┌─────────────────────────────────┬─────────────────────────────────────┐
  │        FIR (order=101)          │       IIR Butterworth (order=5)     │
  ├─────────────────────────────────┼─────────────────────────────────────┤
  │ ✅ Pha tuyến tính (linear)      │ ❌ Pha phi tuyến (nonlinear)       │
  │ ✅ Luôn ổn định (always stable) │ ⚠️  Cần kiểm tra (|pᵢ| < 1)      │
  │ ✅ Bảo toàn hình dạng tín hiệu │ ❌ Méo hình dạng (phase distort)   │
  │ ❌ Nhiều hệ số (101 taps)       │ ✅ Ít hệ số (11 taps)             │
  │ ❌ Transition band rộng hơn     │ ✅ Transition band hẹp hơn        │
  └─────────────────────────────────┴─────────────────────────────────────┘

  → Chọn FIR vì cần bảo toàn temporal shape cho non-stationary class
    (gun_shot, dog_bark — hình dạng xung là thông tin quan trọng)
```

### Tại sao Pipeline B làm 3 bước này?

| Bước | Vấn đề | Công thức | Căn cứ phân tích |
|------|--------|-----------|------------------|
| **FIR Bandpass** | Noise ngoài dải tần hữu ích | $h[n]$: passband $50$–$10000$ Hz | PSD: 99.9% năng lượng < 10304 Hz |
| **Pre-emphasis** | Phổ nghiêng $-6$ dB/octave | $y[n] = x[n] - 0.97 \cdot x[n\!-\!1]$ | Amplitude stats: phổ không phẳng |
| **Normalize** | RMS chênh $30\times$ | $\hat{y} = y / \max|y|$ | RMS: engine=0.122, children=0.004 |

---

## 5. Feature Extraction — Vector 931 chiều

```
                              ┌─────────────────────┐
                              │   Audio Signal x[n]  │
                              │   (88200,)           │
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────┬┴───────────────────┐
                    │                    │                     │
                    ▼                    ▼                     ▼
  ┌──────────────────────────┐ ┌───────────────────┐ ┌────────────────────┐
  │   MFCC Pipeline          │ │ Spectral Features │ │ Spectral Contrast  │
  │                          │ │                   │ │                    │
  │   librosa.feature.mfcc() │ │ • centroid  (1,T) │ │ 7 frequency bands  │
  │   n_mfcc=40              │ │ • bandwidth (1,T) │ │ → (7, T)           │
  │   → MFCC:  (40, T)      │ │ • rolloff   (1,T) │ │                    │
  │                          │ │ • flatness  (1,T) │ │   aggregate_stats  │
  │   librosa.feature.delta  │ │ • ZCR       (1,T) │ │   per band:        │
  │   → Δ:     (40, T)      │ │ • RMS       (1,T) │ │   7 bands × 7 stats│
  │                          │ │                   │ │                    │
  │   librosa.feature.delta  │ │   aggregate_stats │ │   = 49 features    │
  │   order=2                │ │   per feature:     │ │                    │
  │   → Δ²:    (40, T)      │ │   6 feat × 7 stats│ │                    │
  │                          │ │                   │ │                    │
  │   vstack → (120, T)     │ │   = 42 features   │ │                    │
  │                          │ │                   │ │                    │
  │   aggregate_stats        │ │                   │ │                    │
  │   per coefficient:       │ │                   │ │                    │
  │   120 coeff × 7 stats    │ │                   │ │                    │
  │                          │ │                   │ │                    │
  │   = 840 features         │ │                   │ │                    │
  └────────────┬─────────────┘ └─────────┬─────────┘ └──────────┬─────────┘
               │                         │                      │
               └────────────┬────────────┴──────────────────────┘
                            │
                            ▼
               ┌─────────────────────────────┐
               │     np.concatenate()         │
               │                              │
               │   840 + 42 + 49 = 931 dim    │
               │   f ∈ ℝ⁹³¹                   │
               └─────────────────────────────┘
```

### Công thức MFCC

**Bước 1** — Mel spectrogram (mel filterbank × power spectrum):

$$S_\text{mel}(m, t) = \sum_{k} |X(k, t)|^2 \cdot W_m(k), \quad m = 1, \ldots, 128$$

Thang Mel:

$$m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)$$

**Bước 2** — MFCC (DCT trên log mel spectrogram):

$$c_i(t) = \sum_{m=1}^{M} \log\bigl(S_\text{mel}(m, t)\bigr) \cdot \cos\!\left[\frac{\pi i}{M}\left(m - \frac{1}{2}\right)\right], \quad i = 1, \ldots, 40$$

**Bước 3** — Delta (đạo hàm bậc 1):

$$\Delta c_i(t) = \frac{\sum_{\theta=1}^{\Theta} \theta \bigl(c_i(t+\theta) - c_i(t-\theta)\bigr)}{2\sum_{\theta=1}^{\Theta} \theta^2}$$

### 7 Statistical Aggregation Functions

Mỗi feature sequence $\{v_1, v_2, \ldots, v_T\}$ ($T$ frames) → 7 con số cố định:

$$\mu = \frac{1}{T}\sum_{t=1}^{T} v_t \quad \text{(mean)}$$

$$\sigma = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(v_t - \mu)^2} \quad \text{(std)}$$

$$v_\text{min} = \min_{t} v_t, \quad v_\text{max} = \max_{t} v_t$$

$$\tilde{v} = \text{median}(v_1, \ldots, v_T)$$

$$\gamma_1 = \frac{1}{T}\sum_{t=1}^{T}\!\left(\frac{v_t - \mu}{\sigma}\right)^{3} \quad \text{(skewness — độ lệch)}$$

$$\gamma_2 = \frac{1}{T}\sum_{t=1}^{T}\!\left(\frac{v_t - \mu}{\sigma}\right)^{4} - 3 \quad \text{(kurtosis — độ nhọn)}$$

```
  ┌─────────────┬──────────────────────────────────────────────────────┐
  │ Statistic   │ Ý nghĩa cho classification                         │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ mean        │ Giá trị trung bình → đặc trưng "tổng thể"          │
  │ std         │ Độ biến thiên → stationary (thấp) vs non-stat (cao) │
  │ min         │ Giá trị nhỏ nhất → đáy tín hiệu                    │
  │ max         │ Giá trị lớn nhất → đỉnh tín hiệu                   │
  │ median      │ Trung vị → robust hơn mean với outlier              │
  │ skewness    │ Độ lệch phân phối → đối xứng hay không              │
  │ kurtosis    │ Độ nhọn → Gaussian hay có xung (gun_shot)           │
  └─────────────┴──────────────────────────────────────────────────────┘
```

---

## 6. CNN-2D Architecture — Mel Spectrogram Path

```
  Input: Mel Spectrogram (B, 1, 128, 345)
  ════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────┐
  │  Conv Block 1                                                    │
  │  Conv2d(1→32, 3×3, pad=1) → BatchNorm2d(32) → ReLU             │
  │  → MaxPool2d(2×2)                                               │
  │  Output: (B, 32, 64, 172)                                       │
  └────────────────────────────────┬─────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼─────────────────────────────────┐
  │  Conv Block 2                                                    │
  │  Conv2d(32→64, 3×3, pad=1) → BatchNorm2d(64) → ReLU            │
  │  → MaxPool2d(2×2)                                               │
  │  Output: (B, 64, 32, 86)                                        │
  └────────────────────────────────┬─────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼─────────────────────────────────┐
  │  Conv Block 3                                                    │
  │  Conv2d(64→128, 3×3, pad=1) → BatchNorm2d(128) → ReLU          │
  │  → MaxPool2d(2×2)                                               │
  │  Output: (B, 128, 16, 43)                                       │
  └────────────────────────────────┬─────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼─────────────────────────────────┐
  │  Conv Block 4                                                    │
  │  Conv2d(128→256, 3×3, pad=1) → BatchNorm2d(256) → ReLU         │
  │  → AdaptiveAvgPool2d(1×1)                                       │
  │  Output: (B, 256, 1, 1)                                         │
  └────────────────────────────────┬─────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼─────────────────────────────────┐
  │  Classifier                                                      │
  │                                                                  │
  │  Flatten ────→ (B, 256)                                          │
  │       │                                                          │
  │  Dropout(0.5)                                                    │
  │       │                                                          │
  │  Linear(256→128) + ReLU                                          │
  │       │                                                          │
  │  Dropout(0.3)                                                    │
  │       │                                                          │
  │  Linear(128→10)                                                  │
  │       │                                                          │
  │  Output: (B, 10) logits ────→ softmax → class predictions       │
  └──────────────────────────────────────────────────────────────────┘
```

### Training Configuration

| Parameter | Value | Công thức |
|-----------|-------|-----------|
| Optimizer | Adam | $\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ |
| Learning rate | $\eta = 10^{-3}$ | |
| Loss | CrossEntropy | $\mathcal{L} = -\sum_{c=1}^{10} y_c \log(\hat{y}_c)$ |
| LR Scheduler | ReduceLROnPlateau | patience=5, factor=0.5: $\eta \leftarrow 0.5 \cdot \eta$ |
| Early stopping | patience=10 | Stop if val\_loss không giảm 10 epoch |
| Batch size | 32 | |
| Max epochs | 100 | |

### Training Loop

```
  For each fold i ∈ {1, 2, ..., 10}:
  ┌─────────────────────────────────────────────────────────────┐
  │  Train set: 80% of non-test folds                           │
  │  Val set:   20% of non-test folds                           │
  │  Test set:  fold i                                          │
  │                                                             │
  │  For epoch = 1 to 100:                                      │
  │  ┌─────────────────────────────────────────────────────┐    │
  │  │  Forward pass on train batches (batch_size=32)       │    │
  │  │  Loss = CrossEntropyLoss(ŷ, y)                       │    │
  │  │  Backward pass + Adam optimizer step                 │    │
  │  │  Evaluate on val set → val_loss, val_acc             │    │
  │  │                                                      │    │
  │  │  if val_loss không giảm 5 epoch:                     │    │
  │  │      lr ← lr × 0.5                                  │    │
  │  │  if val_loss không giảm 10 epoch:                    │    │
  │  │      EARLY STOP                                      │    │
  │  └─────────────────────────────────────────────────────┘    │
  │                                                             │
  │  Load best weights → Predict on test fold → Store metrics   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 7. Evaluation Pipeline

```
           Pipeline A                              Pipeline B
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │ SVM:  [a₁, a₂, ..., a₁₀]│          │ SVM:  [b₁, b₂, ..., b₁₀]│
  │ RF:   [a₁, a₂, ..., a₁₀]│          │ RF:   [b₁, b₂, ..., b₁₀]│
  │ CNN:  [a₁, a₂, ..., a₁₀]│          │ CNN:  [b₁, b₂, ..., b₁₀]│
  └────────────┬─────────────┘          └────────────┬─────────────┘
               │                                     │
               └──────────────┬──────────────────────┘
                              │
                              ▼
               ┌──────────────────────────────────────┐
               │   Compute Differences per fold        │
               │   dᵢ = bᵢ − aᵢ,  i = 1, ..., 10     │
               └──────────────────┬───────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
  ┌──────────────────┐ ┌─────────────────┐ ┌──────────────────┐
  │  Paired t-test    │ │  Wilcoxon       │ │  Cohen's d       │
  │                   │ │  Signed-rank    │ │  Effect size     │
  │       d̄           │ │  (non-param)    │ │                  │
  │  t = ─────        │ │                 │ │       d̄          │
  │      s_d/√N       │ │  Rank-based     │ │  d = ─────       │
  │                   │ │  comparison     │ │      s_d         │
  │  H₀: μ_d = 0     │ │                 │ │                  │
  │  → p-value        │ │  → p-value      │ │  |d|<0.2: negl.  │
  └────────┬─────────┘ └────────┬────────┘ └────────┬─────────┘
           │                    │                    │
           └────────────┬───────┴────────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   p < 0.05 ?    │
               └────┬───────┬────┘
                    │       │
              Yes   │       │  No
                    ▼       ▼
           ┌──────────┐  ┌──────────────────────────────────┐
           │ DSP giúp  │  │ DSP KHÔNG cải thiện accuracy     │
           │ ích!      │  │ (kết quả thực tế: p = 0.63–0.85)│
           └──────────┘  └──────────────────────────────────┘
```

### Công thức kiểm định thống kê

**Paired $t$-test:**

$$t = \frac{\bar{d}}{s_d / \sqrt{N}}, \quad \bar{d} = \frac{1}{N}\sum_{i=1}^{N} d_i, \quad s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}$$

trong đó $d_i = b_i - a_i$, $N = 10$ folds.

**Cohen's $d$ effect size:**

$$d = \frac{\bar{d}}{s_d}, \quad |d| < 0.2 \text{ (negligible)}, \quad 0.2 \leq |d| < 0.5 \text{ (small)}, \quad 0.5 \leq |d| < 0.8 \text{ (medium)}$$

**95% Confidence Interval:**

$$\text{CI}_{95\%} = \bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{N}}$$

### Kết quả thực tế

| Model | Pipeline A (Raw) | Pipeline B (DSP) | $\Delta$ (B−A) | $p$-value | Cohen's $d$ | Sig.? |
|-------|-----------------|------------------|---------|---------|-----------|----------|
| **SVM** | $70.1 \pm 3.2\%$ | $70.0 \pm 3.6\%$ | $-0.12\%$ | 0.8464 | negligible | No |
| **Random Forest** | $71.5 \pm 2.6\%$ | $71.2 \pm 2.2\%$ | $-0.26\%$ | 0.7680 | negligible | No |
| **CNN-2D** | $66.7 \pm 5.2\%$ | $67.6 \pm 4.8\%$ | $+0.94\%$ | 0.6252 | negligible | No |

---

## 8. Cấu trúc Source Code & Dependencies

```
                    ┌─────────────────────────────────────────┐
                    │          config.py                       │
                    │  TARGET_SR, N_FFT, N_MFCC, FMIN, FMAX   │
                    │  FIR_ORDER, SVM_C_RANGE, CNN_EPOCHS      │
                    │  CLASS_NAMES, RANDOM_SEED, paths...       │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────┬───────────┼───────────┬────────────┐
              │            │           │           │            │
              ▼            ▼           ▼           ▼            ▼
  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
  │ data_loader  │ │signal_       │ │dsp_pipeline  │ │feature_      │
  │              │ │analysis      │ │              │ │extraction    │
  │ load_meta()  │ │              │ │ design_fir() │ │              │
  │ load_audio() │ │ compute_fft()│ │ pre_emph()   │ │ extract_     │
  │ get_fold()   │ │ compute_psd()│ │ normalize()  │ │  mfcc()      │
  │              │ │ compute_cwt()│ │ pipeline_b() │ │ extract_     │
  │              │ │ estimate_    │ │              │ │  spectral()  │
  │              │ │  snr()       │ │              │ │ extract_     │
  │              │ │              │ │              │ │  mel()       │
  └──────┬───────┘ └──────────────┘ └──────┬──────┘ └──────┬───────┘
         │                                 │               │
         │              ┌──────────────────┘               │
         │              │                                  │
         │              ▼                                  ▼
         │   ┌──────────────────────────────────────────────────┐
         │   │              models/                              │
         │   │  ┌─────────────────────┐  ┌────────────────────┐ │
         │   │  │  classical_ml.py    │  │  deep_learning.py  │ │
         │   │  │  • train_svm()      │  │  • CNN2D class     │ │
         │   │  │  • train_rf()       │  │  • train_cnn()     │ │
         │   │  │  • GridSearchCV     │  │  • PyTorch         │ │
         │   │  └──────────┬──────────┘  └──────────┬─────────┘ │
         │   └─────────────┼────────────────────────┼───────────┘
         │                 │                        │
         │                 └────────────┬───────────┘
         │                              │
         │                              ▼
         │                   ┌────────────────────┐
         │                   │  evaluation.py      │
         │                   │  compute_metrics()  │
         │                   │  compare_pipelines()│
         │                   │  paired_t_test()    │
         │                   │  cohens_d()         │
         │                   └─────────┬──────────┘
         │                             │
         │                             ▼
         │                   ┌────────────────────┐
         │                   │  visualization.py   │
         │                   │  plot_waveforms()   │
         │                   │  plot_confusion()   │
         │                   │  plot_roc_curves()  │
         │                   └────────────────────┘
         │
         ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                    Notebooks (thứ tự chạy)                    │
  │                                                               │
  │   pre_analyze ──→ 01_signal ──→ 02_dsp ──→ 03_feature        │
  │       │            analysis      pipeline    engineering      │
  │       │               │             │            │            │
  │       │               ▼             ▼            ▼            │
  │       │          Phân tích     Thiết kế     Feature cache     │
  │       │          tín hiệu     bộ lọc       (2.3 GB)          │
  │       │                                                       │
  │   ──→ 04_pipeline_a ──→ 05_pipeline_b ──→ 06_comparative     │
  │       (Raw models)       (DSP models)      analysis           │
  │           │                  │                  │              │
  │           ▼                  ▼                  ▼              │
  │       results_a.pkl     results_b.pkl    Statistical tests    │
  │       SVM: 70.1%        SVM: 70.0%       p-values, Cohen's d │
  │       RF:  71.5%        RF:  71.2%       → Report + Slides   │
  │       CNN: 66.7%        CNN: 67.6%                            │
  └───────────────────────────────────────────────────────────────┘
```

---

## 9. Từ phân tích → quyết định → code

```
  PHÂN TÍCH (Notebook 01)              QUYẾT ĐỊNH (config.py)           CODE (src/)
  ══════════════════════               ════════════════════              ════════════

  PSD + Cumulative Energy ───────────→ TARGET_SR = 22050 ──────────→ data_loader.py
    f_max(99.9%) = 10304 Hz             (Nyquist ≥ 10000 Hz)          sr=22050

  PSD + Bandwidth (90%,99.9%) ───────→ FILTER = 50–10000 Hz ──────→ dsp_pipeline.py
    Tín hiệu hữu ích: 50–10kHz          FMIN=50, FMAX=10000          FIR bandpass
    Dưới 50 Hz = DC offset

  Window Size Comparison ────────────→ N_FFT = 2048 ──────────────→ feature_extraction.py
    2048: Δt=93ms, Δf=10.7Hz            HOP_LENGTH = 512             n_fft=2048

  Spectral Leakage ──────────────────→ WINDOW = hann ─────────────→ feature_extraction.py
    Hann: sidelobe −31 dB                                             window='hann'

  Stationarity (CV_RMS) ────────────→ Features = 931-dim ─────────→ feature_extraction.py
    6 stat + 4 non-stat                  MFCC+Δ+Δ²+spectral          extract_handcrafted()
    Cần temporal dynamics                + 7 stats aggregation

  SNR Analysis ──────────────────────→ 2 Pipelines: A vs B ───────→ dsp_pipeline.py
    Một số class SNR thấp                để so sánh DSP effect        pipeline_b_process()

  Amplitude Statistics ──────────────→ Normalize + Pre-emphasis ──→ dsp_pipeline.py
    RMS chênh 30× giữa class            trong Pipeline B             normalize + pre_emph

  STFT vs CWT ───────────────────────→ FIR order=101 ────────────→ dsp_pipeline.py
    Temporal shape quan trọng            Linear phase, bảo toàn       design_fir_bandpass()
    cho non-stationary class             hình dạng tín hiệu
```

---

## 10. Bảng tham chiếu nhanh

### Hyperparameters → Nguồn gốc

| Parameter | Giá trị | File | Phân tích nguồn |
|-----------|---------|------|-----------------|
| `TARGET_SR` | $22050$ Hz | config.py:22 | PSD: Nyquist $\geq 10000$ Hz |
| `FILTER_LOW_FREQ` | $50$ Hz | config.py:28 | $< 50$ Hz = DC offset |
| `FILTER_HIGH_FREQ` | $10000$ Hz | config.py:29 | $f_\text{high}^{99.9\%} = 10304$ Hz |
| `FIR_ORDER` | $101$ | config.py:30 | Linear phase, bảo toàn temporal shape |
| `PRE_EMPHASIS_COEFF` | $0.97$ | config.py:32 | Cân bằng phổ ($-6$ dB/octave) |
| `N_FFT` | $2048$ | config.py:35 | $\Delta t = 93$ ms, $\Delta f = 10.7$ Hz |
| `HOP_LENGTH` | $512$ | config.py:36 | $= N_\text{FFT}/4$, overlap 75% |
| `WINDOW_TYPE` | hann | config.py:38 | Sidelobe $-31$ dB, chuẩn librosa |
| `N_MFCC` | $40$ | config.py:41 | Chi tiết cao cho 10 class |
| `N_MELS` | $128$ | config.py:42 | Đủ phân giải mel |
| `FMIN` | $50$ Hz | config.py:43 | $=$ `FILTER_LOW_FREQ` |
| `FMAX` | $10000$ Hz | config.py:44 | $=$ `FILTER_HIGH_FREQ` |

### Source files → Chức năng

| File | Chức năng chính | Được gọi bởi |
|------|----------------|-------------|
| `config.py` | Tất cả hyperparameters | Mọi file |
| `src/data_loader.py` | Load audio, chia fold | Notebook 01–06 |
| `src/signal_analysis.py` | FFT, PSD, STFT, CWT, DWT, SNR | Notebook 01, pre\_analyze |
| `src/dsp_pipeline.py` | FIR/IIR, pre-emphasis, normalize | Notebook 02, 05 |
| `src/feature_extraction.py` | MFCC, spectral, mel spectrogram | Notebook 03–05 |
| `src/models/classical_ml.py` | SVM, Random Forest + GridSearch | Notebook 04, 05 |
| `src/models/deep_learning.py` | CNN-2D (PyTorch) | Notebook 04, 05 |
| `src/evaluation.py` | Metrics, $t$-test, Wilcoxon, Cohen's $d$ | Notebook 06 |
| `src/visualization.py` | Tất cả biểu đồ | Notebook 01–06 |

---

## Related Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design tổng quan
- [MODEL_SPEC.md](MODEL_SPEC.md) — Chi tiết kiến trúc model
- [TECH_STACK.md](TECH_STACK.md) — Dependencies
- [../analyze.md](../analyze.md) — Chi tiết phân tích tín hiệu
