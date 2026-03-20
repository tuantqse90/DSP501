# Signal Analysis — UrbanSound8K (10 Classes)

## 1. Signal Type: Stationary vs Non-Stationary

### Method

Classify each class using **CV_RMS** (Coefficient of Variation of RMS):

$$CV = \frac{\sigma_{\text{RMS}}}{\mu_{\text{RMS}}}$$

- Chia tín hiệu 4s thành 8 đoạn (0.5s/đoạn), tính RMS mỗi đoạn
- **CV < 0.3 → Stationary** (năng lượng ổn định theo thời gian)
- **CV ≥ 0.3 → Non-stationary** (năng lượng thay đổi theo thời gian)

### Results

| Class | CV_RMS | Type | Waveform Pattern | Crest Factor |
|-------|--------|------|------------------|--------------|
| engine_idling | 0.025 | **Stationary** | Ù đều, biên độ ổn định suốt 4s | 2.67 (thấp nhất) |
| air_conditioner | 0.052 | **Stationary** | Noise đều, không có xung nhọn | 7.72 |
| street_music | 0.080 | **Stationary** | Biên độ khá đều, có nhịp nhẹ | 5.19 |
| jackhammer | 0.083 | **Stationary** | Rung lặp đều, chu kỳ rõ | 4.81 |
| siren | 0.142 | **Stationary** | Quét tần số liên tục, biên độ ổn định | 4.79 |
| children_playing | 0.177 | **Stationary** | Rất nhỏ, đều, noise-like | 6.54 |
| drilling | 0.518 | **Non-stationary** | Có lúc khoan lúc dừng, on-off pattern | 5.36 |
| gun_shot | 1.393 | **Non-stationary** | 1 xung nhọn rồi im lặng hoàn toàn | 13.99 |
| car_horn | 1.923 | **Non-stationary** | Vài tiếng bóp ngắn xen kẽ im lặng | 10.83 |
| dog_bark | 2.646 | **Non-stationary** | Vài cụm sủa nhọn, phần lớn im lặng | **21.28** (cao nhất) |

### Interpretation

```
Stationary (6 classes)                   Non-stationary (4 classes)
─────────────────────                    ──────────────────────────
• Phổ tần số KHÔNG ĐỔI theo thời gian   • Phổ tần số THAY ĐỔI theo thời gian
• Waveform amplitude ổn định             • Có xung nhọn / im lặng xen kẽ
• Crest Factor thấp (2.7–7.7)           • Crest Factor cao (5.4–21.3)
• Feature: MFCC mean đã đủ              • Feature: Cần delta MFCC + delta²
• Phân tích: FFT/PSD đủ tốt             • Phân tích: Cần STFT/CWT/spectrogram
```

### Impact on System Design

| Design Decision | Stationary Classes | Non-stationary Classes |
|----------------|-------------------|----------------------|
| Feature extraction | MFCC mean captures spectral shape | Delta MFCC (đạo hàm bậc 1) + Delta² (bậc 2) bắt temporal dynamics |
| Statistical aggregation | Mean, median đủ | Std, skew, kurtosis quan trọng để mã hóa biến động |
| Time-frequency analysis | STFT đủ tốt | CWT tốt hơn (adaptive resolution) |
| Filter choice | IIR hoặc FIR đều OK | **FIR bắt buộc** — linear phase bảo toàn hình dạng xung |

→ **Quyết định**: Feature vector 931-dim kết hợp cả MFCC + delta + delta² + 7 statistical moments → phục vụ cả 2 nhóm.

---

## 2. Frequency Characteristics

### 2.1 Dominant Frequencies (từ PSD Welch, trung bình 30 mẫu/class)

| Class | Top 3 Dominant Freq (Hz) | Physical Explanation | Spectral Type |
|-------|--------------------------|---------------------|---------------|
| engine_idling | **22**, 32, 54 | Harmonics tần số quay động cơ (~1320 RPM) | Narrowband, tonal |
| jackhammer | **75**, 129, 140 | Tần số đập búa khí nén | Broadband |
| air_conditioner | **118**, 108, 97 | Quạt + máy nén hoạt động | Broadband, noise-like |
| street_music | **215**, 226, 97 | Nhạc cụ tần số thấp (bass, drums) | Medium band |
| car_horn | **323**, 334, 345 | Tần số thiết kế còi xe | Medium band |
| dog_bark | **851**, 1637, 1701 | Cộng hưởng thanh quản chó | Medium band |
| siren | **861**, 829, 807 | Thiết kế để tai người nghe rõ (1–4 kHz) | Narrowband, tonal |
| drilling | **1863**, 1949, 1809 | Tần số quay mũi khoan tốc độ cao | Medium band |
| children_playing | 54, 22, 43 | Background noise tần số thấp | Broadband |
| gun_shot | 22, 32, 11 | Broadband impulse, dominant ở DC/rất thấp | Broadband |

**Key Insight**: Mỗi class có "fingerprint tần số" khác nhau rõ ràng → MFCC (mã hóa hình dạng phổ) là feature phù hợp cho phân loại.

### 2.2 Bandwidth Analysis (90% Energy)

| Class | f_low (Hz) | f_high (Hz) | Bandwidth (Hz) | Type |
|-------|-----------|-------------|----------------|------|
| engine_idling | 22 | 248 | **226** | **Narrowband** — năng lượng tập trung, dễ nhận diện |
| dog_bark | 431 | 2,412 | 1,981 | Medium |
| siren | 269 | 2,261 | 1,992 | Medium |
| street_music | 205 | 2,412 | 2,207 | Medium |
| gun_shot | 86 | 2,713 | 2,627 | Medium |
| drilling | 517 | 3,725 | 3,208 | Medium |
| car_horn | 194 | 3,488 | 3,295 | Medium |
| children_playing | 345 | 4,102 | 3,758 | Medium-broad |
| air_conditioner | 43 | 4,210 | 4,167 | **Broadband** — trải rộng, giống noise |
| jackhammer | 65 | 6,772 | **6,708** | **Broadband** — rộng nhất |

### 2.3 Energy Distribution by Frequency Band (DWT)

```
Frequency Band     0────86 Hz    86────345 Hz   345────689 Hz   689────1378 Hz   1378────2756 Hz   2756────5512 Hz
                   (A8 band)     (D8+D7+D6)     (D6 band)      (D5 band)        (D4 band)         (D3 band)

engine_idling      ████████ 79%
gun_shot           ████ 36%
street_music                                    ███ 33%
car_horn                                                        ████ 37%
jackhammer                                                                       ██ 19%
siren                                                                            ██████ 61%
drilling                                                                                           █████ 52%
dog_bark                                                                                           █████ 47%
air_conditioner                                                                                    ███ 27%
children_playing                                                                                   ███ 27%
```

### 2.4 Classes dễ nhầm lẫn (spectral overlap)

| Pair | Reason | How to Distinguish |
|------|--------|--------------------|
| engine_idling ↔ air_conditioner | Cả hai stationary, năng lượng tần số thấp | Engine: narrowband (226 Hz), AC: broadband (4167 Hz) |
| dog_bark ↔ drilling | Cùng dominant band D3 (2756–5512 Hz) | Dog: impulsive (CF=21), Drilling: semi-continuous (CF=5.4) |
| children_playing ↔ street_music | Cả hai broadband, overlap 345–2412 Hz | Children: rất nhỏ (RMS=0.004), Street: to hơn 12× (RMS=0.049) |
| gun_shot ↔ car_horn | Cả hai non-stationary, xung nhọn | Gun: 1 xung duy nhất, Car_horn: nhiều xung lặp lại |

→ **Impact**: Confusion matrix sẽ cho thấy error tập trung ở các cặp này — đây là giới hạn của spectral features, cần thêm temporal features (delta MFCC, ZCR pattern) để phân biệt.

---

## 3. Noise Sources

### 3.1 Types of Noise in UrbanSound8K

| Noise Type | Source | Affected Classes | Frequency Range |
|-----------|--------|-----------------|-----------------|
| **DC offset** | Microphone bias, recording equipment | Tất cả (mức độ khác nhau) | 0–50 Hz |
| **Low-freq rumble** | Gió, rung cơ khí, xe cộ qua lại | air_conditioner, engine_idling, children_playing | < 50 Hz |
| **High-freq noise** | Nhiễu điện tử, quantization, hiss | Tất cả (nhỏ) | > 10,000 Hz |
| **Background ambient** | Tiếng ồn môi trường đô thị (traffic, people) | children_playing, street_music | Broadband |
| **Impulsive noise** | Clicks, pops từ microphone | Ngẫu nhiên | Broadband, ngắn |
| **Natural spectral tilt** | Đặc tính vật lý âm thanh tự nhiên | Tất cả | Tần số thấp mạnh hơn ~6 dB/octave |

### 3.2 SNR Estimation

**Method**: Frame-based (25ms frames, 10th percentile power = noise floor)

$$\text{SNR (dB)} = 10 \cdot \log_{10}\frac{P_{\text{signal}}}{P_{\text{noise}}}$$

| Class | SNR (dB) | Interpretation | DSP Impact |
|-------|---------|----------------|------------|
| air_conditioner | **2.0** | Tín hiệu gần như KHÔNG PHÂN BIỆT được với noise | DSP giúp: +4.5 dB |
| engine_idling | **2.9** | Tương tự — continuous noise-like signal | DSP giúp ít |
| siren | 3.4 | SNR thấp nhưng tonal rõ → MFCC vẫn bắt được | DSP giúp ít |
| jackhammer | 3.8 | Broadband nhưng có pattern lặp → nhận diện được | DSP giúp: +2.9 dB |
| children_playing | 4.9 | Tín hiệu rất nhỏ, noise nền lớn | DSP giúp: +4.5 dB |
| street_music | 5.8 | Trung bình | DSP giúp ít |
| drilling | **24.6** | Tín hiệu rất rõ, noise nhỏ | DSP KHÔNG CẦN |
| car_horn | **∞** | Có đoạn im lặng hoàn toàn (P_noise = 0) | Không áp dụng |
| dog_bark | **∞** | Có đoạn im lặng hoàn toàn | Không áp dụng |
| gun_shot | **∞** | Xung nhọn + im lặng hoàn toàn | Không áp dụng |

### 3.3 How DSP Pipeline Addresses Each Noise Type

```
Noise Source              │  DSP Solution                  │  Filter/Method
─────────────────────────┼────────────────────────────────┼──────────────────────────
DC offset (0 Hz)          │  Bandpass filter (f_low=50 Hz) │  FIR: loại bỏ hoàn toàn
Low-freq rumble (<50 Hz)  │  Bandpass filter (f_low=50 Hz) │  FIR: loại bỏ hoàn toàn
High-freq noise (>10 kHz) │  Bandpass filter (f_high=10kHz)│  FIR: loại bỏ hoàn toàn
Spectral tilt (~6 dB/oct) │  Pre-emphasis filter           │  y[n] = x[n] - 0.97·x[n-1]
Amplitude variation       │  Peak normalization            │  x_out = x / max(|x|)
```

### 3.4 Why DSP Didn't Improve Classification (Key Finding)

Mặc dù DSP cải thiện SNR (+2.9 đến +4.5 dB), accuracy KHÔNG cải thiện đáng kể:

1. **Dataset đã sạch sẵn**: UrbanSound8K là dataset curated — noise đã ít sẵn
2. **MFCC/Mel = implicit DSP**: MFCC dùng `FMIN=50, FMAX=10000` → đã tự "lọc" giống bandpass
3. **ML tự học bỏ noise**: Random Forest / SVM tự tìm feature hữu ích, bỏ qua noise dimensions
4. **Bandpass gần identity**: 99%+ năng lượng đã nằm trong 50–10,000 Hz → filter gần như không thay đổi tín hiệu

---

## 4. Sampling Requirements

### 4.1 Sample Rate: 22,050 Hz

**Derivation from data:**

```
Step 1: Tính PSD (Welch) trung bình 30 mẫu/class → tìm f_high tại ngưỡng 99.9% năng lượng
Step 2: f_max(99.9%) = 10,304 Hz (jackhammer — class có dải tần rộng nhất)
Step 3: Nyquist theorem:  sr ≥ 2 × f_max  →  sr ≥ 2 × 10,000 = 20,000 Hz
Step 4: Chọn sr = 22,050 Hz (> 20,000 Hz ✓, margin 10%)
```

| Candidate SR | Nyquist | Verdict |
|-------------|---------|---------|
| 8,000 Hz | 4,000 Hz | ❌ Mất jackhammer (6772 Hz), drilling harmonics |
| 16,000 Hz | 8,000 Hz | ❌ Mất jackhammer tail, gun_shot broadband |
| **22,050 Hz** | **11,025 Hz** | **✅ Bao phủ 99.9% năng lượng mọi class** |
| 44,100 Hz | 22,050 Hz | ⚠️ Dư thừa, tốn 2× RAM (~6GB vs 3GB), không thêm thông tin |

**Lý do chọn 22,050 Hz cụ thể:**
- = 44,100 / 2 → downsample từ CD quality dễ dàng (chia 2)
- Là default của librosa → tương thích tốt, chuẩn ngành audio ML
- Nyquist = 11,025 Hz > 10,304 Hz (f_max 99.9%) → không aliasing

### 4.2 Audio Duration: 4 seconds = 88,200 samples

```
N_SAMPLES = TARGET_SR × AUDIO_DURATION = 22,050 × 4.0 = 88,200 samples/clip
```

- UrbanSound8K clips đã ≤ 4 seconds → pad ngắn hơn, truncate dài hơn
- 4s đủ dài để bắt pattern lặp (dog_bark, jackhammer cycles)
- 4s đủ ngắn để giữ RAM hợp lý (8732 clips × 88,200 × 4 bytes ≈ 3 GB)

### 4.3 Time-Frequency Resolution (Heisenberg Uncertainty Trade-off)

$$\Delta t \times \Delta f \geq \frac{1}{4\pi}$$

| Window Size (N_FFT) | Δt (ms) | Δf (Hz) | Pros | Cons |
|---------------------|---------|---------|------|------|
| 512 | 23 ms | 43 Hz | Thời gian tốt → bắt gun_shot | Không phân biệt engine (22 Hz) vs jackhammer (75 Hz) |
| 1024 | 46 ms | 21.5 Hz | Trung bình | Trung bình |
| **2048** | **93 ms** | **10.7 Hz** | **Cân bằng tốt nhất** | |
| 4096 | 186 ms | 5.4 Hz | Tần số chi tiết | Mất timing gun_shot (xung ~50ms < window 186ms) |

**Chọn N_FFT = 2048:**
- Δf = 10.7 Hz → phân biệt engine (22 Hz) vs jackhammer (75 Hz) ✅
- Δt = 93 ms → bắt transient dog_bark, gun_shot ✅
- Lũy thừa 2 → FFT tính toán nhanh nhất (Cooley-Tukey)

### 4.4 Hop Length & Overlap

```
HOP_LENGTH = N_FFT / 4 = 2048 / 4 = 512 samples
Overlap = 1 - (512/2048) = 75%
```

- 75% overlap: chuẩn ngành audio processing
- Mel spectrogram shape: **(128 mel bands × 345 time frames)** per clip
- 345 frames = (88,200 - 2048) / 512 + 1

### 4.5 Mel Scale Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| N_MELS | 128 | Đủ phân giải cho 10 class trong dải 50–10,000 Hz |
| FMIN | 50 Hz | = FILTER_LOW_FREQ, loại DC offset |
| FMAX | 10,000 Hz | = FILTER_HIGH_FREQ, bao phủ 99.9% năng lượng |
| N_MFCC | 40 | Nhiều hơn standard (13) → capture thêm chi tiết phổ |

**Mel scale formula:**

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

128 mel filters trải đều từ mel(50) = 77 đến mel(10000) = 3817 → mật độ cao ở tần số thấp (nơi tai người nhạy nhất), thưa ở tần số cao.

### 4.6 Window Function: Hann

| Window | Sidelobe Level | Mainlobe Width | Verdict |
|--------|---------------|----------------|---------|
| Boxcar | -13 dB (nhiều leakage) | Hẹp nhất | ❌ Spectral leakage gây "tần số giả" trong MFCC |
| **Hann** | **-31 dB** | **Vừa** | **✅ Trade-off tốt nhất** |
| Hamming | -42 dB | Vừa | OK nhưng Hann là librosa default |
| Blackman | -58 dB (ít leakage nhất) | Rộng nhất | ❌ Mainlobe rộng → mất chi tiết tần số |

### 4.7 Summary: All Sampling Parameters

```
┌──────────────────────────────────────────────────────────────┐
│                  SAMPLING CONFIGURATION                      │
├──────────────────────┬───────────────────────────────────────┤
│ TARGET_SR            │ 22,050 Hz                             │
│ AUDIO_DURATION       │ 4.0 seconds                           │
│ N_SAMPLES            │ 88,200 samples/clip                   │
│ N_FFT                │ 2,048 samples                         │
│ HOP_LENGTH           │ 512 samples (overlap 75%)             │
│ WIN_LENGTH           │ 2,048 samples                         │
│ WINDOW_TYPE          │ hann                                  │
│ N_MELS               │ 128 mel filter banks                  │
│ N_MFCC               │ 40 coefficients                       │
│ FMIN                 │ 50 Hz                                 │
│ FMAX                 │ 10,000 Hz                             │
│ FILTER_LOW_FREQ      │ 50 Hz                                 │
│ FILTER_HIGH_FREQ     │ 10,000 Hz                             │
│ FIR_ORDER            │ 101 taps                              │
│ PRE_EMPHASIS_COEFF   │ 0.97                                  │
├──────────────────────┴───────────────────────────────────────┤
│ Mel spectrogram shape: (128, 345) per clip                   │
│ Feature vector: 931 dimensions                               │
│ Total dataset in RAM: ~3 GB (88,200 × 8,732 × float32)      │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: 10 Classes at a Glance

| Class | Stationary? | Dominant Freq | Bandwidth (90%) | SNR | Key Characteristic |
|-------|------------|---------------|-----------------|-----|-------------------|
| engine_idling | ✅ Yes | 22 Hz | 226 Hz (narrow) | 2.9 dB | Ù trầm đều, harmonics rõ |
| air_conditioner | ✅ Yes | 118 Hz | 4,167 Hz (broad) | 2.0 dB | Noise đều, giống white noise |
| jackhammer | ✅ Yes | 75 Hz | 6,708 Hz (broad) | 3.8 dB | Rung lặp đều, bandwidth rộng nhất |
| street_music | ✅ Yes | 215 Hz | 2,207 Hz (medium) | 5.8 dB | Nhạc cụ tần số thấp |
| siren | ✅ Yes | 861 Hz | 1,992 Hz (medium) | 3.4 dB | Tonal, quét tần số, pattern đặc trưng |
| children_playing | ✅ Yes | 54 Hz | 3,758 Hz (medium) | 4.9 dB | Rất nhỏ (RMS=0.004), noise-like |
| drilling | ❌ No | 1,863 Hz | 3,208 Hz (medium) | 24.6 dB | On-off pattern, SNR cao nhất |
| gun_shot | ❌ No | 22 Hz | 2,627 Hz (medium) | ∞ | 1 xung nhọn + im lặng |
| car_horn | ❌ No | 323 Hz | 3,295 Hz (medium) | ∞ | Vài tiếng bóp ngắn |
| dog_bark | ❌ No | 851 Hz | 1,981 Hz (medium) | ∞ | Xung nhọn nhất (CF=21.3) |
