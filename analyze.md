# Phân tích tín hiệu — UrbanSound8K

## Mục đích

Phân tích đặc tính tín hiệu của 10 class âm thanh môi trường để đưa ra các quyết định thiết kế hệ thống:
- Chọn sample rate
- Thiết kế bộ lọc bandpass
- Cấu hình feature extraction (MFCC, Mel spectrogram)
- Dự đoán class nào khó phân loại

Kết quả phân tích từ `notebooks/01_signal_analysis.ipynb` và `notebooks/pre_analyze_signal.ipynb`.

---

## 0. Tổng quan — Mỗi phần phân tích trả lời câu hỏi gì?

| Phần | Câu hỏi cần trả lời | Quyết định rút ra |
|---|---|---|
| **Class Distribution** | Dataset có cân bằng không? | car_horn (429), gun_shot (374) ít hơn → cần lưu ý khi đánh giá |
| **Waveform** | Tín hiệu trông như thế nào? Dừng hay không dừng? | Biết class nào stationary (dễ) vs non-stationary (khó) |
| **FFT / PSD** | Năng lượng tập trung ở tần số nào? | → Chọn `sr=22050`, `FILTER=50-10000 Hz`, `FMIN/FMAX` cho MFCC |
| **Dominant Frequencies** | Mỗi class có tần số đặc trưng gì? | Hiểu cơ sở vật lý: engine=22Hz, siren=1130Hz... |
| **Mel Spectrogram** | Spectrogram khác nhau thế nào giữa các class? | Xác nhận mel spectrogram là input tốt cho CNN |
| **Window Size** | Window nào cho trade-off tốt nhất? | → Chọn `N_FFT=2048` (cân bằng thời gian vs tần số) |
| **SNR** | Tín hiệu sạch hay nhiễu? | Biết class nào cần DSP nhất (SNR thấp) |
| **Stationarity** | Tín hiệu có thay đổi theo thời gian? | Stationary → feature tần số đủ; Non-stationary → cần time-frequency |
| **Spectral Leakage** | Window function nào giảm leakage tốt nhất? | → Chọn `window='hann'` (trade-off tốt) |
| **Wavelet (CWT/DWT)** | STFT hay CWT tốt hơn cho dữ liệu này? | CWT tốt hơn cho non-stationary (gun_shot), STFT đủ cho stationary |

### Chi tiết từng phần:

### 0.1. Class Distribution — Dataset có cân bằng không?

**Phân tích:** Đếm số lượng mẫu mỗi class trong metadata.

**Kết quả:**
- Phần lớn class có ~1000 mẫu
- **car_horn: chỉ 429 mẫu** (ít hơn 57% so với class đầy đủ)
- **gun_shot: chỉ 374 mẫu** (ít hơn 63%)

**Quyết định rút ra:**
- Không dùng accuracy đơn thuần → cần thêm **F1-score (macro)** để đánh giá công bằng
- Macro average tính F1 cho từng class rồi lấy trung bình → class ít mẫu có trọng số bằng class nhiều mẫu
- Không cần oversampling/undersampling vì cross-validation 10-fold của UrbanSound8K đã được thiết kế để mỗi fold có tỷ lệ class tương đối cân bằng

**Nếu bỏ qua:**
- Nếu chỉ nhìn accuracy: Model có thể đạt 87.5% chỉ bằng cách LUÔN đoán class có 1000 mẫu → tưởng tốt nhưng thực ra car_horn/gun_shot bị dự đoán sai hoàn toàn

---

### 0.2. Waveform — Tín hiệu trông như thế nào?

**Phân tích:** Vẽ biên độ theo thời gian cho mỗi class, tính RMS, Peak, Crest Factor.

**Câu hỏi cần trả lời:**
- Tín hiệu có **đều đặn** (stationary) hay **thay đổi** (non-stationary)?
- **Biên độ** lớn hay nhỏ? Có xung nhọn không?

**Kết quả:**
- engine_idling, air_conditioner: Waveform đều đặn suốt 4 giây → stationary
- gun_shot: 1 xung nhọn rồi im lặng → rõ ràng non-stationary
- dog_bark: Vài cụm biên độ cao xen kẽ im lặng → non-stationary
- Crest Factor dao động 2.67 (engine) → 21.28 (dog_bark) → xác nhận nhận xét trực quan

**Quyết định rút ra:**
- Phân biệt sơ bộ 2 nhóm class → định hướng cho việc chọn feature (mục 0.8)
- Class có CF cao (xung nhọn) → cần feature bắt temporal pattern (delta MFCC, ZCR)
- Class có CF thấp (đều) → feature tần số tĩnh (MFCC mean) đã đủ

**Giới hạn:** Waveform KHÔNG cho biết tín hiệu chứa tần số nào → cần FFT/PSD

---

### 0.3. FFT / PSD — Năng lượng tập trung ở tần số nào?

**Phân tích:**
- FFT: Chuyển tín hiệu sang miền tần số (1 phép tính, noisy)
- PSD (Welch): Ước lượng phổ ổn định hơn (trung bình nhiều đoạn)
- Bandwidth: Tìm dải tần chứa 90% / 99% / 99.9% năng lượng
- Trung bình PSD trên 30 mẫu/class → dải tần đáng tin

**Câu hỏi cần trả lời:**
- Mỗi class có năng lượng ở dải tần nào?
- Tần số cao nhất cần giữ lại là bao nhiêu?

**Kết quả:**
- f_high (99.9%) max = 10304 Hz → tất cả tín hiệu hữu ích nằm dưới ~10000 Hz
- f_low min = 22 Hz → nhưng dưới 50 Hz chủ yếu là DC offset

**Quyết định rút ra:**
- `TARGET_SR = 22050` (Nyquist >= 10000 Hz)
- `FILTER_LOW_FREQ = 50`, `FILTER_HIGH_FREQ = 10000`
- `FMIN = 50`, `FMAX = 10000` (cho MFCC và Mel spectrogram)

Đây là phần phân tích **quan trọng nhất**, quyết định 3 hyperparameters cùng lúc. Chi tiết đầy đủ xem mục 9.1, 9.2, 9.3.

---

### 0.4. Dominant Frequencies — Mỗi class có tần số đặc trưng gì?

**Phân tích:** Từ PSD, sắp xếp theo năng lượng giảm dần, lấy top 5 tần số.

**Câu hỏi cần trả lời:**
- Tần số nào **đặc trưng** cho mỗi loại âm thanh?
- Có thể giải thích bằng vật lý không?

**Kết quả:**
- engine_idling: 22 Hz → tần số quay động cơ (~1320 RPM)
- siren: 861 Hz → tần số còi hú (thiết kế cho tai người)
- jackhammer: 75 Hz → tần số đập búa khí nén
- dog_bark: 851 Hz → cộng hưởng thanh quản chó

**Quyết định rút ra:**
- Xác nhận rằng mỗi class có "fingerprint tần số" riêng → **có thể phân loại bằng feature tần số**
- Nếu tất cả class có cùng dominant frequency → phân loại bằng tần số sẽ thất bại → phải tìm feature khác
- Kết quả cho thấy dominant freq KHÁC NHAU đáng kể → MFCC (mã hóa hình dạng phổ) là feature phù hợp

---

### 0.5. Mel Spectrogram — Spectrogram khác nhau thế nào giữa các class?

**Phân tích:** Tính mel spectrogram (128 mels × T frames) cho mỗi class, visualize dưới dạng ảnh.

**Câu hỏi cần trả lời:**
- Mel spectrogram có **khác biệt trực quan** giữa các class không?
- Có phù hợp làm input cho CNN-2D không?

**Kết quả:**
- engine_idling: Dải sáng ở tần số thấp, đều suốt thời gian
- siren: Dải sáng quét lên xuống ở tần số giữa (pattern đặc trưng)
- gun_shot: Cột sáng dọc (toàn dải tần) tại thời điểm nổ
- dog_bark: Vài cột sáng tại thời điểm sủa

**Quyết định rút ra:**
- Mel spectrogram cho **pattern trực quan KHÁC NHAU rõ ràng** giữa các class → **phù hợp làm input cho CNN-2D**
- CNN-2D xử lý mel spectrogram giống xử lý "ảnh" → có thể học pattern tự động
- Xác nhận thiết kế: Pipeline dùng cả **handcrafted features (931-dim)** cho ML truyền thống VÀ **mel spectrogram** cho CNN

---

### 0.6. Window Size — Window nào cho trade-off tốt nhất?

**Phân tích:** So sánh STFT spectrogram với window sizes 512, 1024, 2048, 4096 trên cùng tín hiệu (siren).

**Câu hỏi cần trả lời:**
- Trade-off thời gian vs tần số (Heisenberg uncertainty): Window nào cân bằng tốt nhất cho dataset này?

**Kết quả:**
- 512: Thấy rõ timing nhưng tần số mờ
- 4096: Thấy rõ tần số nhưng timing mờ
- **2048: Cả hai đều chấp nhận được** — Δt=93ms (đủ bắt transient), Δf=10.7 Hz (đủ phân biệt harmonics)

**Quyết định rút ra:**
- `N_FFT = 2048`, `WIN_LENGTH = 2048`
- `HOP_LENGTH = 512` (= N_FFT/4 → overlap 75%, chuẩn ngành)

Chi tiết xem mục 9.4.

---

### 0.7. SNR — Tín hiệu sạch hay nhiễu?

**Phân tích:** Ước lượng SNR bằng cách chia frame thành signal (mạnh) vs noise (yếu).

**Câu hỏi cần trả lời:**
- Class nào có tín hiệu "sạch"? Class nào bị nhiễu?
- DSP preprocessing có thể cải thiện class nào?

**Kết quả:**
- **SNR rất thấp (2-3 dB):** air_conditioner, engine_idling → tín hiệu giống noise, khó tách
- **SNR cao (24.6 dB):** drilling → tín hiệu rõ ràng, DSP không cần thiết
- **SNR = ∞:** car_horn, dog_bark, gun_shot → có đoạn im lặng hoàn toàn

**Quyết định rút ra:**
- DSP (bandpass filter) có thể giúp class SNR thấp bằng cách loại bỏ noise ngoài dải → tăng SNR
- Đây là **lý do thiết kế 2 pipeline** (A: Raw vs B: DSP) để kiểm chứng giả thuyết
- Kết quả cuối cùng: DSP tăng SNR +2.9 đến +4.5 dB nhưng accuracy KHÔNG cải thiện đáng kể (p > 0.05)

Chi tiết xem mục 9.7.

---

### 0.8. Stationarity — Tín hiệu có thay đổi theo thời gian?

**Phân tích:** Tính CV_RMS (coefficient of variation) qua 8 đoạn 0.5 giây.

**Câu hỏi cần trả lời:**
- Class nào có tín hiệu **dừng** (stationary — phổ không đổi theo thời gian)?
- Class nào **không dừng** (non-stationary — phổ thay đổi)?

**Kết quả:**
- **6 class stationary** (CV < 0.3): engine_idling, air_conditioner, jackhammer, street_music, siren, children_playing
- **4 class non-stationary** (CV ≥ 0.3): drilling, gun_shot, car_horn, dog_bark

**Quyết định rút ra:**
- Stationary class: MFCC mean (trung bình theo thời gian) đủ để mô tả → feature tần số đơn giản
- Non-stationary class: Cần **delta MFCC** (đạo hàm bậc 1) và **delta2** (bậc 2) để bắt SỰ THAY ĐỔI phổ theo thời gian
- Cần **statistical aggregation** (7 stats: mean, std, min, max, median, skew, kurtosis) để mã hóa cả behavior ổn định (mean) lẫn biến động (std, kurtosis)
- → Feature vector 931-dim kết hợp cả tần số + temporal dynamics

Chi tiết xem mục 9.6.

---

### 0.9. Spectral Leakage — Window function nào giảm leakage tốt nhất?

**Phân tích:** So sánh FFT spectrum với 4 window functions: boxcar, hann, hamming, blackman.

**Câu hỏi cần trả lời:**
- Window nào giảm "tần số giả" (spectral leakage) tốt nhất mà không mất chi tiết?

**Kết quả:**
- Boxcar: Leakage nhiều nhất (sidelobe -13 dB) nhưng mainlobe hẹp nhất
- Blackman: Leakage ít nhất (-58 dB) nhưng mainlobe rộng nhất
- **Hann: Trade-off tốt** (sidelobe -31 dB, mainlobe vừa)

**Quyết định rút ra:**
- `WINDOW_TYPE = "hann"` — cũng là default của librosa
- Spectral leakage ảnh hưởng trực tiếp đến chất lượng MFCC: nếu leakage nhiều → MFCC chứa "tần số giả" → feature không chính xác

Chi tiết xem mục 9.5.

---

### 0.10. Wavelet (CWT/DWT) — STFT hay CWT tốt hơn?

**Phân tích:**
- CWT (Continuous Wavelet Transform): Scalogram với Morlet wavelet
- DWT (Discrete Wavelet Transform): Phân tách năng lượng theo dải tần (multi-resolution)
- So sánh STFT vs CWT trên 3 class đại diện

**Câu hỏi cần trả lời:**
- Phương pháp time-frequency nào phù hợp hơn cho dataset này?
- DWT cho thêm thông tin gì mà PSD không có?

**Kết quả:**
- CWT bắt gun_shot (transient) tốt hơn STFT (phân giải thời gian cao ở tần số cao)
- STFT đủ tốt cho stationary class (air_conditioner)
- DWT cho biết **dải tần chủ đạo** rõ hơn PSD: engine_idling = 78.9% ở 0-86 Hz, siren = 60.6% ở 1378-2756 Hz

**Quyết định rút ra:**
- Dự án dùng **STFT-based features** (MFCC, mel spectrogram) vì đơn giản hơn và librosa hỗ trợ tốt
- CWT/DWT dùng để **phân tích và hiểu dữ liệu**, không dùng trực tiếp làm feature
- DWT energy giúp xác nhận dải tần dominant → validate kết quả PSD
- DWT cũng cho thấy cặp class overlap (dog_bark & drilling cùng band D3) → dự đoán confusion matrix

Chi tiết filter type xem mục 9.10.

---

## 1. Thông tin dataset

| Thông số | Giá trị |
|---|---|
| Tổng số mẫu | 8732 clips |
| Số class | 10 |
| Số folds | 10 (predefined, không shuffle) |
| Sample rate | 22050 Hz |
| Thời lượng mỗi clip | 4 giây (88200 samples) |
| Mono | Có (1 kênh) |

---

## 2. Các thuật ngữ kỹ thuật

### RMS (Root Mean Square) — Năng lượng trung bình

$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

- Bình phương từng mẫu → trung bình → căn bậc 2
- Đo "âm lượng trung bình" của tín hiệu
- RMS cao = âm thanh to, liên tục; RMS thấp = âm thanh nhỏ
- Không dùng mean thường vì tín hiệu audio dao động quanh 0 → mean ≈ 0

### Peak — Biên độ cực đại

$$\text{Peak} = \max_n |x[n]|$$

- Giá trị biên độ lớn nhất (trị tuyệt đối) trong toàn bộ tín hiệu

### Crest Factor — Tỷ lệ đỉnh nhọn

$$CF = \frac{\text{Peak}}{\text{RMS}}$$

- Đo "hình dạng" tín hiệu: năng lượng đều đặn hay có xung nhọn
- CF thấp (2–5): năng lượng đều → engine_idling, jackhammer, siren
- CF cao (10–21): xung nhọn, phần lớn im lặng → dog_bark, gun_shot, car_horn
- CF = √2 ≈ 1.414 cho sóng sin thuần (trường hợp lý tưởng)

### CV_RMS (Coefficient of Variation of RMS) — Độ ổn định

$$CV = \frac{\sigma_{\text{RMS}}}{\mu_{\text{RMS}}}$$

- Chia tín hiệu thành 8 đoạn (0.5 giây/đoạn), tính RMS mỗi đoạn
- CV = std(RMS các đoạn) / mean(RMS các đoạn)
- CV < 0.3 → Stationary (năng lượng ổn định theo thời gian)
- CV ≥ 0.3 → Non-stationary (năng lượng thay đổi theo thời gian)

### Dominant Frequency — Tần số mạnh nhất

- Tần số có năng lượng (PSD) lớn nhất trong phổ tần số
- Tính PSD (Welch) → sắp xếp giảm dần → lấy tần số ứng với PSD lớn nhất
- Phản ánh tần số đặc trưng vật lý: engine=22Hz (tần số quay), siren=1130Hz (thiết kế cho tai người)

### Bandwidth (90%) — Dải tần chứa 90% năng lượng

- Tính cumulative energy từ 0 Hz → Nyquist
- f_low = tần số tại mốc tích lũy 5%
- f_high = tần số tại mốc tích lũy 95%
- Bandwidth = f_high − f_low
- Narrowband (< 1500 Hz): tín hiệu tonal, tập trung
- Broadband (> 4000 Hz): tín hiệu trải rộng, giống noise

### SNR (Signal-to-Noise Ratio) — Tỷ lệ tín hiệu / nhiễu

$$\text{SNR (dB)} = 10 \cdot \log_{10}\frac{P_{\text{signal}}}{P_{\text{noise}}}$$

- Chia thành frame 25ms, lấy 10% frame yếu nhất làm noise floor
- SNR = 0 dB → signal = noise; SNR = 20 dB → signal gấp 100× noise
- SNR = ∞ → có frame im lặng hoàn toàn (car_horn, dog_bark, gun_shot)

---

## 3. Kết quả phân tích — Amplitude Statistics

| Class | RMS | Peak | Crest Factor | Nhận xét |
|---|---|---|---|---|
| engine_idling | 0.1222 | 0.3257 | 2.67 | Năng lượng cao, rất đều |
| drilling | 0.1215 | 0.6506 | 5.36 | Năng lượng cao, hơi biến động |
| air_conditioner | 0.0972 | 0.7500 | 7.72 | Năng lượng khá, noise-like |
| street_music | 0.0491 | 0.2547 | 5.19 | Trung bình |
| dog_bark | 0.0463 | 0.9859 | **21.28** | Xung cực nhọn, phần lớn im lặng |
| siren | 0.0453 | 0.2168 | 4.79 | Đều, tonal |
| jackhammer | 0.0323 | 0.1555 | 4.81 | Đều, rung lặp |
| gun_shot | 0.0264 | 0.3697 | **13.99** | Xung nhọn rồi im lặng |
| car_horn | 0.0174 | 0.1880 | **10.83** | Vài tiếng bóp ngắn |
| children_playing | 0.0041 | 0.0271 | 6.54 | Rất nhỏ |

---

## 4. Kết quả phân tích — Stationarity & SNR

| Class | CV_RMS | Stationarity | SNR (dB) |
|---|---|---|---|
| engine_idling | 0.025 | Stationary | 2.9 |
| air_conditioner | 0.052 | Stationary | 2.0 |
| street_music | 0.080 | Stationary | 5.8 |
| jackhammer | 0.083 | Stationary | 3.8 |
| siren | 0.142 | Stationary | 3.4 |
| children_playing | 0.177 | Stationary | 4.9 |
| drilling | 0.518 | **Non-stationary** | 24.6 |
| gun_shot | 1.393 | **Non-stationary** | ∞ |
| car_horn | 1.923 | **Non-stationary** | ∞ |
| dog_bark | 2.646 | **Non-stationary** | ∞ |

**Nhận xét:**
- 6 class stationary: năng lượng ổn định → FFT/PSD đủ để mô tả
- 4 class non-stationary: năng lượng thay đổi → cần STFT/spectrogram/wavelet
- SNR = ∞ cho car_horn, dog_bark, gun_shot vì có đoạn im lặng hoàn toàn (P_noise = 0)
- SNR thấp (2-3 dB) cho air_conditioner, engine_idling vì tín hiệu giống noise

---

## 5. Kết quả phân tích — Tần số

### Dominant Frequencies (từ PSD, 1 mẫu/class)

| Class | Top 3 Dominant Frequencies | Giải thích vật lý |
|---|---|---|
| engine_idling | 22, 32, 54 Hz | Harmonics tần số quay động cơ |
| jackhammer | 75, 129, 140 Hz | Tần số đập búa khí nén |
| air_conditioner | 118, 108, 97 Hz | Quạt + máy nén |
| street_music | 215, 226, 97 Hz | Nhạc cụ tần số thấp |
| car_horn | 323, 334, 345 Hz | Tần số còi xe |
| dog_bark | 851, 1637, 1701 Hz | Cộng hưởng thanh quản chó |
| siren | 861, 829, 807 Hz | Tần số còi hú (thiết kế cho tai người) |
| drilling | 1863, 1949, 1809 Hz | Tần số quay mũi khoan |
| children_playing | 54, 22, 43 Hz | Noise tần số thấp (background) |
| gun_shot | 22, 32, 11 Hz | Broadband, dominant ở DC/rất thấp |

### DWT Energy Distribution (top band mỗi class)

| Class | Top Band | Dải tần | % Năng lượng |
|---|---|---|---|
| engine_idling | A8 | 0–86 Hz | **78.9%** |
| siren | D4 | 1378–2756 Hz | **60.6%** |
| drilling | D3 | 2756–5512 Hz | **51.5%** |
| dog_bark | D3 | 2756–5512 Hz | **46.6%** |
| car_horn | D5 | 689–1378 Hz | **36.5%** |
| gun_shot | A8 | 0–86 Hz | 35.7% |
| street_music | D6 | 345–689 Hz | 33.4% |
| air_conditioner | D3 | 2756–5512 Hz | 26.8% |
| children_playing | D3 | 2756–5512 Hz | 26.8% |
| jackhammer | D4 | 1378–2756 Hz | 19.4% |

---

## 6. Kết quả phân tích — Bandwidth

### Bandwidth 50% năng lượng (từ 1 mẫu/class)

| Class | f_low | f_high | Bandwidth |
|---|---|---|---|
| engine_idling | 22 Hz | 32 Hz | 11 Hz |
| street_music | 118 Hz | 355 Hz | 237 Hz |
| siren | 818 Hz | 1130 Hz | 312 Hz |
| gun_shot | 22 Hz | 463 Hz | 441 Hz |
| car_horn | 345 Hz | 958 Hz | 614 Hz |
| dog_bark | 883 Hz | 1712 Hz | 829 Hz |
| drilling | 1314 Hz | 2283 Hz | 969 Hz |
| jackhammer | 269 Hz | 1755 Hz | 1486 Hz |
| children_playing | 75 Hz | 2433 Hz | 2358 Hz |
| air_conditioner | 151 Hz | 2595 Hz | 2444 Hz |

### Bandwidth 90% năng lượng (trung bình 30 mẫu/class)

| Class | f_low (Hz) | f_high (Hz) | Bandwidth (Hz) | Dominant (Hz) | Type |
|---|---|---|---|---|---|
| engine_idling | 22 | 248 | 226 | 22 | Narrowband |
| dog_bark | 431 | 2412 | 1981 | 484 | Medium |
| siren | 269 | 2261 | 1992 | 1130 | Medium |
| street_music | 205 | 2412 | 2207 | 463 | Medium |
| gun_shot | 86 | 2713 | 2627 | 323 | Medium |
| drilling | 517 | 3725 | 3208 | 1830 | Medium |
| car_horn | 194 | 3488 | 3295 | 2347 | Medium |
| children_playing | 345 | 4102 | 3758 | 1270 | Medium |
| air_conditioner | 43 | 4210 | 4167 | 118 | Broadband |
| jackhammer | 65 | 6772 | 6708 | 75 | Broadband |

### So sánh f_high ở các ngưỡng năng lượng

| Class | 90% | 95% | 99% | 99.9% |
|---|---|---|---|---|
| engine_idling | 248 Hz | 1109 Hz | 2239 Hz | 6901 Hz |
| siren | 2261 Hz | 2498 Hz | 4544 Hz | 9884 Hz |
| street_music | 2412 Hz | 2907 Hz | 4963 Hz | 8280 Hz |
| dog_bark | 2412 Hz | 2929 Hz | 5599 Hz | 9636 Hz |
| gun_shot | 2713 Hz | 4210 Hz | 8226 Hz | 10164 Hz |
| car_horn | 3488 Hz | 3650 Hz | 5534 Hz | 8398 Hz |
| drilling | 3725 Hz | 4877 Hz | 8032 Hz | 10293 Hz |
| children_playing | 4102 Hz | 5416 Hz | 8032 Hz | 9981 Hz |
| air_conditioner | 4210 Hz | 5071 Hz | 8387 Hz | 9916 Hz |
| jackhammer | 6772 Hz | 7763 Hz | 9485 Hz | 10304 Hz |
| **MAX (toàn dataset)** | **6772 Hz** | **7763 Hz** | **9485 Hz** | **10304 Hz** |

**Tại sao chọn f_high = 10000 Hz khi bảng 90% chỉ cho max 6772 Hz?**
- Ở ngưỡng 99%: f_max = 9485 Hz → vẫn còn năng lượng đáng kể trên 6772 Hz
- Ở ngưỡng 99.9%: f_max = 10304 Hz → gần như toàn bộ năng lượng
- Các class broadband (air_conditioner, jackhammer, drilling) có "đuôi" phổ kéo dài
- 10000 Hz là số tròn, chuẩn ngành, và giữ thừa tốt hơn cắt thiếu

---

## 7. Class nào dễ nhầm với nhau?

| Cặp dễ nhầm | Lý do |
|---|---|
| engine_idling ↔ air_conditioner | Cả hai stationary, năng lượng tần số thấp |
| dog_bark ↔ drilling | DWT: cùng dominant ở band D3 (2756–5512 Hz) |
| children_playing ↔ street_music | Cả hai broadband, dải tần overlap 345–2412 Hz |
| gun_shot ↔ car_horn | Cả hai non-stationary, xung nhọn, crest factor cao |

---

## 8. STFT vs CWT

| Thuộc tính | STFT | CWT (Morlet) |
|---|---|---|
| Phân giải | Cố định (phụ thuộc window size) | Thích ứng theo tần số |
| Tần số cao | Phân giải thời gian kém | Phân giải thời gian tốt |
| Tần số thấp | Phân giải tần số kém | Phân giải tần số tốt |
| Tốt cho | Stationary (air_conditioner) | Non-stationary (gun_shot, siren) |
| Tính toán | Nhanh hơn (FFT-based) | Chậm hơn |

---

## 9. Luồng logic: Từ phân tích → quyết định

Mỗi phân tích trong notebook đều phục vụ cho **ít nhất 1 quyết định thiết kế** cụ thể. Dưới đây là luồng logic chi tiết.

### 9.1. PSD + Bandwidth → Chọn Sample Rate (TARGET_SR = 22050 Hz)

**Phân tích đã làm:**
- Tính PSD (Welch) trung bình trên 30 mẫu/class
- Tính cumulative energy → tìm f_high tại các ngưỡng 90%, 95%, 99%, 99.9%

**Kết quả:**
- f_high (99.9%) max = 10304 Hz (jackhammer)
- Nghĩa là: toàn bộ tín hiệu hữu ích nằm dưới ~10000 Hz

**Lý do chọn:**
- Theo định lý Nyquist: `sr >= 2 * f_max` → `sr >= 2 * 10000 = 20000 Hz`
- 22050 Hz > 20000 Hz → thỏa mãn Nyquist với margin 10%
- 22050 = 44100/2 → downsample từ CD quality rất dễ
- Là default của librosa, chuẩn ngành audio ML

**Nếu chọn sai:**
- sr = 8000 Hz → Nyquist = 4000 Hz → mất drilling (1800 Hz OK nhưng harmonics bị cắt), jackhammer (đến 6772 Hz bị cắt)
- sr = 44100 Hz → Nyquist = 22050 Hz → dư thừa, tốn gấp đôi RAM (6GB thay vì 3GB) mà không thêm thông tin

```
PSD analysis → f_max ≈ 10000 Hz → Nyquist → sr = 22050 Hz
```

---

### 9.2. PSD + Bandwidth → Thiết kế bộ lọc Bandpass (50–10000 Hz)

**Phân tích đã làm:**
- Bảng bandwidth 90% → tìm f_low min và f_high max
- Bảng so sánh ngưỡng 90% vs 99.9% → xác nhận "đuôi" phổ

**Kết quả:**
- f_low nhỏ nhất: engine_idling = 22 Hz (nhưng dưới 50 Hz chủ yếu là DC offset + rung cơ học)
- f_high (99.9%) max: jackhammer = 10304 Hz

**Lý do chọn f_low = 50 Hz (không phải 22 Hz):**
- Dưới 50 Hz: DC offset, rung cơ khí, tiếng gió → KHÔNG phải thông tin âm thanh hữu ích
- Tai người nghe kém dưới 50 Hz → không đóng góp cho phân loại
- Loại bỏ DC offset giúp các phép tính sau (RMS, MFCC) chính xác hơn

**Lý do chọn f_high = 10000 Hz (không phải 6772 Hz):**
- 6772 Hz chỉ là ngưỡng 90% → vẫn còn 10% năng lượng phía trên
- Ở ngưỡng 99%: f_max = 9485 Hz → gần 10000 Hz
- Ở ngưỡng 99.9%: f_max = 10304 Hz → vượt 10000 Hz nhưng sát
- 10000 Hz là số tròn, chuẩn ngành, bao phủ 99%+ năng lượng cho MỌI class

**Nếu chọn sai:**
- f_low = 500 Hz → mất engine_idling (dominant 22 Hz), jackhammer (dominant 75 Hz) → 2 class bị ảnh hưởng
- f_high = 4000 Hz → mất jackhammer (đến 6772 Hz), drilling (đến 3725 Hz tại 90%) → accuracy giảm
- Không lọc (0–11025 Hz) → giữ noise DC + tần số cao → SNR không cải thiện

```
f_low analysis → 50 Hz (loại DC, rung cơ)
f_high analysis → 10000 Hz (bao phủ 99%+ toàn bộ class)
→ Bandpass filter: 50–10000 Hz
```

---

### 9.3. PSD + Bandwidth → Cấu hình Feature Extraction (FMIN=50, FMAX=10000)

**Phân tích đã làm:**
- Cùng kết quả bandwidth ở mục 9.2

**Lý do chọn:**
- MFCC và Mel spectrogram cần biết dải tần để phân bố mel filter banks
- `FMIN=50, FMAX=10000` → 128 mel filters trải đều từ 50 đến 10000 Hz
- Khớp chính xác với dải tần hữu ích từ phân tích PSD

**Nếu chọn sai:**
- FMIN=0, FMAX=11025 (default librosa) → mel filters lãng phí cho vùng 0–50 Hz (DC) và 10000–11025 Hz (noise) → phân giải ở vùng hữu ích giảm ~10%
- FMIN=300, FMAX=8000 → mất engine_idling (dominant 22 Hz ở dưới 300), mất jackhammer "đuôi" phổ

```
Bandwidth analysis → dải tần hữu ích 50–10000 Hz
→ FMIN = 50, FMAX = 10000 (cho MFCC & Mel spectrogram)
→ N_MELS = 128 mel filter banks trải trong dải này
```

---

### 9.4. Window Size Comparison → Chọn N_FFT = 2048

**Phân tích đã làm:**
- So sánh STFT spectrogram với window sizes: 512, 1024, 2048, 4096
- Đánh giá trade-off phân giải thời gian vs tần số (Heisenberg uncertainty)

**Kết quả:**
- Window 512: Phân giải thời gian tốt (23ms) nhưng tần số kém (Δf = 43 Hz)
- Window 1024: Trung bình cả hai
- **Window 2048: Cân bằng tốt** — Δt = 93ms, Δf = 10.7 Hz
- Window 4096: Phân giải tần số tốt (Δf = 5.4 Hz) nhưng thời gian kém (186ms)

**Lý do chọn 2048:**
- Δf = 22050/2048 ≈ 10.7 Hz → đủ phân biệt engine_idling (dominant 22 Hz) với jackhammer (75 Hz)
- Δt = 2048/22050 ≈ 93ms → đủ bắt kịp biến đổi nhanh của gun_shot, dog_bark
- Dataset có cả stationary (cần Δf nhỏ) và non-stationary (cần Δt nhỏ) → 2048 là trade-off tốt nhất
- Là lũy thừa 2 → FFT tính toán nhanh nhất

**Nếu chọn sai:**
- N_FFT = 512: Không phân biệt được engine_idling (22 Hz) vì Δf = 43 Hz > khoảng cách giữa các harmonics
- N_FFT = 4096: Mất chi tiết thời gian của gun_shot (xung ~50ms nhưng window = 186ms → bị "trung bình hóa")

```
Window comparison → 2048 cân bằng time-frequency
→ N_FFT = 2048, WIN_LENGTH = 2048, HOP_LENGTH = 512 (overlap 75%)
```

---

### 9.5. Spectral Leakage Analysis → Chọn Window Function = Hann

**Phân tích đã làm:**
- So sánh spectral leakage của 4 window functions: boxcar, hann, hamming, blackman
- Đo sidelobe level và mainlobe width

**Kết quả:**

| Window | Mainlobe | Sidelobe | Trade-off |
|---|---|---|---|
| Boxcar | Hẹp nhất | Cao nhất (-13 dB) | Phân giải tần số tốt nhưng leakage nhiều |
| **Hann** | **Vừa** | **Thấp (-31 dB)** | **Cân bằng tốt** |
| Hamming | Vừa | Thấp (-42 dB) | Tương tự Hann, sidelobe đầu tiên thấp hơn |
| Blackman | Rộng nhất | Thấp nhất (-58 dB) | Leakage ít nhất nhưng phân giải kém |

**Lý do chọn Hann:**
- Boxcar gây leakage quá nhiều → "tần số giả" ảnh hưởng đến MFCC
- Blackman mainlobe quá rộng → mất chi tiết tần số
- Hann là trade-off tốt, được dùng rộng rãi trong audio processing
- librosa dùng Hann làm default → tương thích tốt

```
Spectral leakage analysis → Hann giảm leakage tốt, mainlobe vừa phải
→ WINDOW_TYPE = "hann"
```

---

### 9.6. Stationarity Analysis → Chọn Feature Strategy (931-dim vector)

**Phân tích đã làm:**
- Tính CV_RMS cho từng class → phân loại stationary vs non-stationary
- So sánh STFT vs CWT trên các class khác nhau

**Kết quả:**
- 6 class stationary (CV < 0.3): Phổ tần số không đổi theo thời gian → chỉ cần feature tần số
- 4 class non-stationary (CV ≥ 0.3): Phổ thay đổi → cần cả time + frequency features

**Lý do chọn 931-dim feature vector kết hợp:**
- **MFCC (840 dim)**: 40 hệ số × 3 (MFCC + delta + delta2) × 7 stats → bắt đặc trưng tần số + SỰ THAY ĐỔI tần số theo thời gian (delta)
- **Spectral features (42 dim)**: centroid, bandwidth, rolloff, flatness, ZCR, RMS × 7 stats → bắt hình dạng phổ + sự thay đổi theo thời gian
- **Spectral contrast (49 dim)**: 7 bands × 7 stats → bắt sự khác biệt peak/valley trong phổ

**Tại sao cần delta và delta2 trong MFCC:**
- MFCC gốc: Chỉ mô tả "phổ tại 1 thời điểm" → đủ cho stationary
- Delta (đạo hàm bậc 1): Mô tả "phổ THAY ĐỔI thế nào" → cần cho non-stationary
- Delta2 (đạo hàm bậc 2): Mô tả "tốc độ thay đổi" → bắt transient (gun_shot, dog_bark)

**Tại sao dùng 7 stats (mean, std, min, max, median, skew, kurtosis):**
- Mean: Giá trị trung bình của feature theo thời gian
- Std: Mức độ biến thiên → stationary (std thấp) vs non-stationary (std cao)
- Min/Max: Giá trị cực trị → bắt xung nhọn
- Skew: Độ lệch phân phối → tín hiệu đối xứng hay không
- Kurtosis: Độ nhọn phân phối → tín hiệu Gaussian hay có outlier (xung)

**Nếu chọn sai:**
- Chỉ dùng MFCC (không delta): Mất thông tin temporal → non-stationary class accuracy giảm
- Chỉ dùng spectral features (42 dim): Quá ít → underfitting
- Không dùng statistical aggregation: Feature vector có kích thước khác nhau giữa các clip → không đưa vào ML được

```
Stationarity analysis → 6 stationary + 4 non-stationary
→ Cần cả frequency features (MFCC) + temporal dynamics (delta)
→ Statistical aggregation (7 stats) → fixed-size 931-dim vector
```

---

### 9.7. SNR Analysis → Đánh giá cần DSP preprocessing không

**Phân tích đã làm:**
- Ước lượng SNR cho từng class
- So sánh SNR trước/sau DSP (trong notebook 02)

**Kết quả:**
- air_conditioner (SNR=2.0 dB), engine_idling (2.9 dB): Tín hiệu gần như noise → DSP có thể giúp
- drilling (24.6 dB): Tín hiệu rất rõ → DSP không cần thiết
- car_horn, dog_bark, gun_shot (SNR=∞): Có đoạn im lặng → DSP tác động không rõ

**Lý do thiết kế Pipeline A vs B:**
- Pipeline A (Raw → Features → Model): Baseline, không DSP
- Pipeline B (Raw → DSP → Features → Model): Thêm bandpass + pre-emphasis + normalize

**Mục đích so sánh:**
- Trả lời câu hỏi nghiên cứu: "DSP preprocessing có cải thiện accuracy không?"
- SNR analysis GỢI Ý rằng DSP có thể giúp class SNR thấp (air_conditioner, engine_idling)
- Nhưng kết quả cuối cùng cho thấy: accuracy không cải thiện đáng kể (p > 0.05)
- Giải thích: ML/DL đã tự học cách "bỏ qua" noise, và bandpass 50–10000 Hz gần như giữ nguyên tín hiệu (vì phần lớn năng lượng đã nằm trong dải này)

```
SNR analysis → Class SNR thấp có thể hưởng lợi từ DSP
→ Thiết kế 2 pipeline (A: Raw, B: DSP) để so sánh
→ Kết quả: DSP không giúp đáng kể (p > 0.05)
```

---

### 9.8. Amplitude Statistics → Quyết định Normalize + Pre-emphasis

**Phân tích đã làm:**
- Tính RMS, Peak, Crest Factor cho từng class

**Kết quả:**
- RMS dao động rất lớn: 0.004 (children_playing) → 0.122 (engine_idling) = chênh 30 lần
- Peak dao động: 0.027 → 0.986

**Lý do cần Normalize trong Pipeline B:**
- Nếu không normalize: Model có thể học "âm lượng" thay vì "nội dung âm thanh"
- Peak normalization: Chia cho max(|x|) → mọi clip có peak = 1.0 → cùng thang đo

**Lý do cần Pre-emphasis trong Pipeline B:**
- Phổ tần số tự nhiên nghiêng: tần số thấp mạnh hơn tần số cao ~6 dB/octave
- Pre-emphasis `y[n] = x[n] - 0.97*x[n-1]` → tăng tần số cao → cân bằng phổ
- Giúp MFCC bắt tốt hơn thông tin ở tần số cao (quan trọng cho drilling, car_horn)

```
Amplitude analysis → RMS chênh 30× giữa các class
→ Normalize (cùng thang đo) + Pre-emphasis (cân bằng phổ)
```

---

### 9.9. DWT Energy + Overlap Analysis → Dự đoán class khó phân loại

**Phân tích đã làm:**
- DWT energy: Xác định band năng lượng chủ đạo cho mỗi class
- PSD overlap: So sánh hình dạng phổ giữa các class

**Kết quả:**
- dog_bark và drilling: Cùng dominant ở band D3 (2756–5512 Hz) → overlap
- air_conditioner và children_playing: Cùng dominant ở band D3, cùng broadband → overlap
- engine_idling và gun_shot: Cùng dominant ở band A8 (0–86 Hz) → overlap

**Mục đích:**
- Dự đoán cặp class nào sẽ bị nhầm trong confusion matrix
- Giúp giải thích kết quả model: Nếu model nhầm dog_bark → drilling, đó là vì phổ tần số giống nhau, KHÔNG phải vì model kém
- Gợi ý hướng cải thiện: Cần feature phân biệt temporal pattern (dog_bark = xung rời rạc, drilling = liên tục)

```
DWT energy overlap → Dự đoán cặp class dễ nhầm
→ Giải thích confusion matrix
→ Gợi ý cần temporal features (delta MFCC, ZCR pattern)
```

---

### 9.10. STFT vs CWT → Chọn FIR filter (bảo toàn pha)

**Phân tích đã làm:**
- So sánh STFT (cố định resolution) vs CWT (adaptive resolution)
- Quan sát trên 3 class: siren (tonal), gun_shot (transient), air_conditioner (noise)

**Kết quả:**
- CWT tốt hơn cho gun_shot (bắt transient chính xác hơn)
- STFT đủ tốt cho stationary class
- Cả hai đều yêu cầu tín hiệu KHÔNG BỊ LỆCH PHA

**Lý do chọn FIR filter (không phải IIR):**
- FIR (order=101): **Pha tuyến tính** → mọi tần số bị trễ đều 50 samples → hình dạng tín hiệu KHÔNG THAY ĐỔI
- IIR (order=5): Pha phi tuyến → tần số khác nhau bị trễ khác nhau → gun_shot bị "méo" hình dạng
- Với gun_shot, dog_bark: Hình dạng xung (temporal shape) là thông tin QUAN TRỌNG → phải bảo toàn

**Nếu chọn IIR:**
- Tính toán nhanh hơn (5 hệ số thay vì 101)
- Nhưng dog_bark có thể bị méo hình dạng xung → MFCC delta bị ảnh hưởng → accuracy giảm cho non-stationary class

```
STFT vs CWT analysis → Temporal shape quan trọng cho non-stationary
→ Chọn FIR (pha tuyến tính, bảo toàn hình dạng tín hiệu)
→ FIR order = 101 (số lẻ → đối xứng hoàn hảo)
```

---

### Tổng hợp: Sơ đồ luồng toàn bộ

```
PHÂN TÍCH (Notebook 01 + pre_analyze)          QUYẾT ĐỊNH (config.py + pipeline)
════════════════════════════════════            ══════════════════════════════════

PSD + Cumulative Energy ──────────────────────→ TARGET_SR = 22050 Hz
  └─ f_max (99.9%) = 10304 Hz                  (Nyquist >= 10000 Hz)
  └─ f_low min = 22 Hz

PSD + Bandwidth (90%, 99.9%) ─────────────────→ FILTER = bandpass 50–10000 Hz
  └─ Tín hiệu hữu ích: 50–10000 Hz            FMIN = 50, FMAX = 10000
  └─ Dưới 50 Hz = DC offset                    (cho MFCC + Mel spectrogram)

Window Size Comparison ───────────────────────→ N_FFT = 2048, HOP_LENGTH = 512
  └─ 2048 cân bằng Δt (93ms) vs Δf (10.7 Hz)

Spectral Leakage ─────────────────────────────→ WINDOW_TYPE = "hann"
  └─ Hann: sidelobe -31 dB, mainlobe vừa

Stationarity (CV_RMS) ────────────────────────→ Feature: 931-dim
  └─ 6 stationary + 4 non-stationary            (MFCC + delta + delta2 + spectral)
  └─ Cần cả frequency + temporal info           + statistical aggregation (7 stats)

SNR Analysis ─────────────────────────────────→ Pipeline A (Raw) vs B (DSP)
  └─ Một số class SNR thấp                      để so sánh hiệu quả DSP

Amplitude Statistics ─────────────────────────→ Normalize + Pre-emphasis
  └─ RMS chênh 30× giữa class                   trong Pipeline B

DWT Energy Overlap ───────────────────────────→ Dự đoán confusion matrix
  └─ Các cặp cùng dominant band                 + giải thích kết quả model

STFT vs CWT ──────────────────────────────────→ FIR filter (order=101)
  └─ Temporal shape quan trọng                   bảo toàn pha tuyến tính
```

---

### Bảng tóm tắt quyết định

| Quyết định | Giá trị | Phân tích nguồn | Lý do |
|---|---|---|---|
| `TARGET_SR` | 22050 Hz | PSD bandwidth | Nyquist >= 10000 Hz, chuẩn librosa |
| `FILTER_LOW_FREQ` | 50 Hz | PSD f_low min | Loại DC offset, dưới 50 Hz không hữu ích |
| `FILTER_HIGH_FREQ` | 10000 Hz | PSD f_high 99.9% | Bao phủ 99%+ năng lượng mọi class |
| `N_FFT` | 2048 | Window comparison | Cân bằng Δt=93ms vs Δf=10.7 Hz |
| `HOP_LENGTH` | 512 | N_FFT / 4 | Overlap 75%, chuẩn audio processing |
| `WINDOW_TYPE` | hann | Spectral leakage | Trade-off mainlobe/sidelobe tốt |
| `FMIN` | 50 | Bandwidth analysis | Khớp với filter low freq |
| `FMAX` | 10000 | Bandwidth analysis | Khớp với filter high freq |
| `N_MELS` | 128 | Dải tần 50–10000 Hz | Đủ phân giải cho 10 class |
| `N_MFCC` | 40 | Chuẩn ngành | 13–40 thường dùng, 40 cho chi tiết cao |
| `FIR_ORDER` | 101 | STFT vs CWT | Số lẻ, pha tuyến tính, transition band chấp nhận được |
| `PRE_EMPHASIS_COEFF` | 0.97 | Amplitude stats | Chuẩn speech processing, cân bằng phổ |
| Feature dim | 931 | Stationarity | MFCC(840) + spectral(42) + contrast(49) |
| Filter type | FIR | Phase analysis | Bảo toàn temporal shape cho non-stationary class |

---

## 10. Tóm tắt đặc tính 10 class

### Nhóm A: Stationary (CV < 0.3)

| Class | Đặc điểm chính |
|---|---|
| **engine_idling** | Tiếng ù trầm rất đều, 78.9% năng lượng dưới 86 Hz, narrowband, dễ nhận diện bằng tần số |
| **air_conditioner** | Noise tần số thấp trải đều, broadband, SNR rất thấp (2.0 dB) |
| **jackhammer** | Rung đều nhưng phổ trải rất rộng (65–6772 Hz), bandwidth lớn nhất |
| **street_music** | Nhiều tần số, dominant ~200–460 Hz, medium bandwidth |
| **siren** | Tonal rõ ~800–1100 Hz, quét tần số, dễ nhận diện |
| **children_playing** | Rất nhỏ (RMS=0.004), noise-like, broadband |

### Nhóm B: Non-stationary (CV ≥ 0.3)

| Class | Đặc điểm chính |
|---|---|
| **drilling** | Có lúc khoan, lúc dừng, dominant ~1800 Hz, SNR cao nhất (24.6 dB) |
| **gun_shot** | Xung nhọn rồi im lặng, broadband, CF=14.0 |
| **car_horn** | Vài tiếng bóp còi ngắn, dominant ~2300 Hz, CF=10.8 |
| **dog_bark** | Vài tiếng sủa nhọn, phần lớn im lặng, CF cao nhất (21.3), CV cao nhất (2.65) |
