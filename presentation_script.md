# Kịch bản thuyết trình — DSP501

> Thời lượng: 12–15 phút | 15 slides
> Checklist đề bài: System block diagram, Filter frequency response, Spectrogram comparison, Performance comparison table, Confusion matrix, Critical interpretation, 6 câu hỏi discussion

---

## Phân bổ thời gian

| Phần | Slides | Thời gian | Trọng số đề bài |
|------|--------|-----------|-----------------|
| Mở đầu + Research Question | 1–2 | 1 phút | — |
| Dataset + Signal Analysis | 3–4 | 2 phút | DSP Design (25%) |
| DSP Pipeline Design | 5–6 | 3 phút | DSP Design (25%) + Before/After (20%) |
| Feature Engineering + Models | 7–8 | 2 phút | AI/ML Design (20%) |
| Evaluation + Results | 9–11 | 3 phút | Experimental Rigor (15%) |
| Discussion + Conclusion | 12–14 | 3 phút | Critical Discussion (10%) |
| Q&A | 15 | 1 phút | — |

---

## SLIDE 1: Title (30 giây)

**Hiển thị:** Tên project, tên nhóm, ngày

**Nói:**

> "Xin chào thầy/cô và các bạn. Nhóm chúng em thực hiện đề tài **Environmental Sound Classification** — phân loại âm thanh môi trường. Câu hỏi nghiên cứu chính: **Tiền xử lý DSP có cải thiện hiệu quả phân loại bằng AI hay không?** Chúng em sử dụng dataset UrbanSound8K với 8732 đoạn âm thanh, 10 loại."

---

## SLIDE 2: Research Question (30 giây)

**Hiển thị:** Câu hỏi nghiên cứu + bảng so sánh Pipeline A vs B

**Nói:**

> "Để trả lời câu hỏi này, chúng em thiết kế **2 pipeline**:
>
> **Pipeline A** — đưa âm thanh thô trực tiếp vào model, không xử lý gì.
>
> **Pipeline B** — trước khi đưa vào model, chúng em áp dụng 3 bước DSP: bộ lọc FIR bandpass, pre-emphasis, và chuẩn hóa biên độ.
>
> Cả hai pipeline đều dùng cùng feature extraction và cùng model — chỉ khác ở bước tiền xử lý. Như vậy, sự khác biệt kết quả **chỉ do DSP** gây ra."

**Lưu ý:** Nhấn mạnh *cùng model, cùng feature → chỉ khác DSP* — thiết kế thí nghiệm công bằng

---

## SLIDE 3: Dataset — UrbanSound8K (1 phút)

**Hiển thị:** Bảng 10 class + `fig_waveforms_per_class.png`

**Nói:**

> "Dataset UrbanSound8K gồm 8732 đoạn âm thanh, mỗi đoạn tối đa 4 giây, chia thành 10 loại âm thanh đô thị. Chúng em resample tất cả về **22050 Hz** vì theo phân tích PSD, 99.9% năng lượng tín hiệu nằm dưới 10304 Hz — tần số Nyquist 11025 Hz đủ đáp ứng.
>
> Một điểm quan trọng: dataset này có **10 predefined folds** — chúng em **không bao giờ shuffle** data across folds, vì các clip từ cùng nguồn gốc được nhóm trong cùng fold. Shuffle sẽ gây **data leakage**.
>
> Chúng em cũng chia signal thành 2 nhóm:
> - **6 class stationary**: air_conditioner, engine_idling... — phổ tần số ổn định theo thời gian
> - **4 class non-stationary**: gun_shot, dog_bark... — có xung nhọn, phổ thay đổi
>
> Sự phân biệt này ảnh hưởng đến cách chúng em thiết kế feature extraction."

**Câu hỏi có thể bị hỏi:**
- *"Tại sao 22050 Hz?"* → PSD: f_max(99.9%)=10304 Hz. Nyquist 11025 Hz > 10304 Hz. 44100 Hz tốn gấp đôi RAM mà không thêm info.
- *"Class imbalance?"* → car_horn (429), gun_shot (374) ít hơn → dùng F1-macro.

---

## SLIDE 4: Signal Analysis (1 phút)

**Hiển thị:** `fig_fft_per_class.png` + `fig_psd_per_class.png`

**Nói:**

> "Trước khi thiết kế bộ lọc, chúng em phân tích đặc tính tín hiệu:
>
> **FFT và PSD** cho thấy mỗi class có 'fingerprint tần số' riêng: engine_idling tập trung ở 22 Hz, siren ở 860 Hz, drilling ở 1860 Hz. Điều này xác nhận rằng **thông tin tần số đủ để phân loại**.
>
> **Spectral leakage**: So sánh 4 loại window — chọn **Hann** vì sidelobe -31 dB — cân bằng tốt.
>
> **Window size**: So sánh N_FFT = 512, 1024, 2048, 4096. Chọn **2048** vì Δt = 93ms và Δf = 10.7 Hz — trade-off tốt nhất cho dataset có cả stationary và non-stationary.
>
> Tất cả quyết định thiết kế đều **có căn cứ từ phân tích tín hiệu**, không chọn tùy ý."

**Câu hỏi có thể bị hỏi:**
- *"Time-frequency trade-off?"* → Heisenberg: window lớn → Δf nhỏ nhưng Δt lớn, và ngược lại.
- *"Welch vs FFT?"* → Welch chia đoạn + overlap + trung bình → giảm variance, ước lượng phổ ổn định hơn.

---

## SLIDE 5: FIR Filter Design (1.5 phút) — TRỌNG TÂM (25%)

**Hiển thị:** Công thức FIR + `fig_fir_frequency_response.png` + `fig_fir_vs_iir_comparison.png`

**Nói:**

> "Pipeline B bắt đầu bằng **FIR bandpass filter**, passband 50–10000 Hz, order 101.
>
> **Tại sao 50–10000 Hz?**
> - Dưới 50 Hz: DC offset và rung cơ khí — không phải âm thanh hữu ích
> - Trên 10000 Hz: f_high(99.9%) = 10304 Hz — gần như không có năng lượng
>
> **Phương pháp**: Window method với Hann window. Đáp ứng xung lý tưởng (vô hạn) nhân với cửa sổ Hann (101 mẫu) → bộ lọc thực tế.
>
> **Tại sao FIR thay vì IIR?**
> - FIR: **pha tuyến tính** → tất cả tần số trễ đều 50 samples → **bảo toàn hình dạng** tín hiệu
> - IIR: pha phi tuyến → tần số khác nhau trễ khác nhau → **méo hình dạng**
>
> Với gun_shot và dog_bark, hình dạng xung là đặc trưng phân loại. Méo → delta MFCC sai → accuracy giảm.
>
> Chúng em dùng **filtfilt** (zero-phase) — lọc xuôi rồi ngược để triệt tiêu phase shift.
>
> IIR Butterworth order 5 cũng được implement để so sánh. Poles nằm trong unit circle → stable. Nhưng phase phi tuyến nên chọn FIR."

**Câu hỏi có thể bị hỏi:**
- *"FIR order 101 nhược điểm?"* → Transition band rộng hơn IIR. 101 phép nhân/sample. Offline processing → OK.
- *"filtfilt vs lfilter?"* → lfilter: có trễ. filtfilt: lọc 2 lần → phase=0, biên độ bị bình phương |H|².
- *"FIR stability?"* → Luôn stable — chỉ có zeros, không có poles.

---

## SLIDE 6: Pre-emphasis + Before/After (1.5 phút) — TRỌNG TÂM (20%)

**Hiển thị:** Công thức pre-emphasis + `fig_before_after_children_playing.png` + SNR table

**Nói:**

> "Sau FIR filter, 2 bước nữa:
>
> **Pre-emphasis**: $y[n] = x[n] - 0.97 \cdot x[n-1]$. Bộ lọc thông cao bậc 1. Phổ tự nhiên nghiêng -6 dB/octave — pre-emphasis bù lại, giúp MFCC bắt tốt tần số cao.
>
> **Peak normalization**: Chia cho giá trị peak → [-1, 1]. RMS giữa các class chênh 30 lần — engine 0.122, children 0.004.
>
> **Kết quả before/after**: DSP cải thiện SNR thực sự:
> - children_playing: +4.5 dB
> - jackhammer: +2.9 dB
>
> Tín hiệu sạch hơn, phổ tập trung hơn. **Nhưng** câu hỏi là: cải thiện SNR có dẫn đến cải thiện accuracy không? Câu trả lời sẽ ở phần kết quả."

**Câu hỏi có thể bị hỏi:**
- *"α = 0.97 từ đâu?"* → Chuẩn speech processing. H(z) = 1 - 0.97z⁻¹ tăng ~20 dB cho tần số cao.
- *"Filtering loại bỏ useful info?"* → (Câu 3 đề bài) Passband 50–10000 Hz bao phủ 99%+. Nhưng engine dominant 22 Hz sát biên dưới → có thể mất một phần.

---

## SLIDE 7: Feature Engineering (1 phút)

**Hiển thị:** Bảng 931-dim + `fig_tsne_features.png`

**Nói:**

> "Feature extraction chuyển mỗi clip (88200 samples) thành vector 931 chiều:
>
> - **840 từ MFCC**: 40 hệ số × 3 (gốc + delta + delta²) × 7 stats = 840
> - **42 spectral**: centroid, bandwidth, rolloff, flatness, ZCR, RMS × 7 stats
> - **49 contrast**: 7 bands × 7 stats
>
> 7 thống kê gồm: mean, std, min, max, median, skewness, kurtosis. Cần cả 7 vì 2 class có thể cùng mean nhưng khác std.
>
> Cho CNN: **mel spectrogram** 128 × 173, cũng fmin=50, fmax=10000.
>
> Điểm quan trọng: MFCC dùng **cùng passband 50–10000 Hz** như FIR filter → sự **trùng lặp**."

**Câu hỏi có thể bị hỏi:**
- *"Tại sao 40 MFCC?"* → 13 là chuẩn speech. Environmental sounds phức tạp hơn → 40 cho chi tiết cao.
- *"Features mathematically justified?"* → Có — MFCC: mel scale, DCT, delta formula. Công thức đầy đủ trong report.

---

## SLIDE 8: Models (1 phút)

**Hiển thị:** 3 models + CNN architecture

**Nói:**

> "3 models — đáp ứng yêu cầu ít nhất 1 classical ML + 1 deep learning:
>
> **SVM**: RBF kernel, PCA 931→200 (vì O(n²d)), GridSearchCV → C=10, γ=scale.
>
> **Random Forest**: 500 cây, full 931 features. Cho ra feature importance.
>
> **CNN-2D**: 4 conv blocks (32→64→128→256), BatchNorm, MaxPool, AdaptiveAvgPool. Adam lr=0.001, early stopping patience=10.
>
> Tất cả dùng **cùng cross-validation protocol và cùng features**."

**Câu hỏi có thể bị hỏi:**
- *"Overfitting?"* → CNN: early stopping + dropout. RF: 500 trees giảm variance. SVM: regularization C.
- *"Hyperparameter tuning?"* → GridSearchCV cho SVM, RF. CNN: ReduceLROnPlateau.

---

## SLIDE 9: Evaluation Methodology (1 phút)

**Hiển thị:** 10-fold CV diagram + statistical tests table

**Nói:**

> "**10-fold CV** predefined, không shuffle. Metrics: accuracy, F1-macro, 95% CI.
>
> So sánh Pipeline A vs B bằng 3 kiểm định:
> - **Paired t-test**: $t = \bar{d} / (s_d / \sqrt{N})$ — mean difference khác 0 không?
> - **Wilcoxon**: non-parametric alternative
> - **Cohen's d**: effect size — |d| < 0.2 = negligible
>
> Ngưỡng α = 0.05. p > 0.05 → không significant."

---

## SLIDE 10: Results (1 phút)

**Hiển thị:** Bảng accuracy + `fig_accuracy_comparison_barplot.png` + Cohen's d

**Nói:**

> "Kết quả chính:
>
> **Random Forest** cao nhất: 71.5% (A) vs 71.2% (B), chênh -0.26%, p = 0.768.
> **SVM**: 70.1% vs 70.0%, chênh -0.12%, p = 0.846.
> **CNN-2D**: 66.7% vs 67.6%, chênh +0.94%, p = 0.625.
>
> Tất cả Cohen's d < 0.2 → **negligible** — không chỉ không significant thống kê, mà còn không significant thực tiễn."

---

## SLIDE 11: Statistical Analysis (1 phút)

**Hiển thị:** `fig_fold_accuracy_boxplot.png`

**Nói:**

> "Box plot cho thấy variance giữa folds **lớn hơn nhiều** so với chênh lệch A vs B.
>
> CNN spread: 53.6%–81.1%. RF ổn định nhất: 62.7%–78.5%.
>
> **Thành phần fold** ảnh hưởng accuracy nhiều hơn DSP. Fold nào có nhiều class khó (engine vs air_conditioner dễ nhầm) → accuracy thấp, bất kể DSP hay không."

---

## SLIDE 12: Why Doesn't DSP Help? (1.5 phút) — CRITICAL DISCUSSION (10%)

**Hiển thị:** 2 lý do chính

**Nói:**

> "Phần discussion quan trọng nhất. Tại sao DSP không cải thiện accuracy?
>
> **Lý do 1: Dataset đã sạch sẵn.** UrbanSound8K là dataset nghiên cứu — đã chọn lọc, cắt gọn, kiểm tra chất lượng. SNR tương đối cao. Bộ lọc FIR loại rất ít vì **không có noise ngoài dải để loại**. Trên dataset thu âm thực tế — mic rẻ, ồn — DSP sẽ giúp ích hơn.
>
> **Lý do 2: Feature extraction đã DSP ngầm.** MFCC dùng fmin=50, fmax=10000 — **cùng passband** FIR filter. StandardScaler = normalize. Statistical aggregation robust với noise = giống filtering.
>
> **Clean data + implicit DSP = explicit DSP redundant.**"

**Trả lời 6 câu hỏi đề bài:**

> "**1. DSP cải thiện không?** → Không significant (p > 0.05).
>
> **2. Frequency bands discriminative?** → Khác theo class: engine 22 Hz, siren 860 Hz, drilling 1860 Hz — chứng minh bằng PSD và DWT.
>
> **3. Filtering loại useful info?** → Rất ít. 50–10000 Hz bao phủ 99%+. Nhưng engine dominant 22 Hz sát biên dưới.
>
> **4. Preprocessing ảnh hưởng overfitting?** → Không rõ ràng — cả 2 pipeline variance tương đương.
>
> **5. Computational complexity?** → FIR: ~81 tỷ phép nhân cho dataset. Thêm ~30 giây. Không đáng kể cho offline.
>
> **6. DSP cần thiết cho deep learning?** → Dataset sạch: không. CNN tự học từ mel spectrogram. Dataset noisy: có thể giúp hội tụ nhanh."

---

## SLIDE 13: Model Comparison (30 giây)

**Hiển thị:** Bảng RF vs SVM vs CNN

**Nói:**

> "**Classical ML > Deep Learning** trên dataset này. RF 71.5% > CNN 66.7%. Nguyên nhân: ~7800 training samples quá ít cho CNN 4 layers. Với data augmentation, CNN có thể cải thiện. RF ổn định nhất và train nhanh nhất."

---

## SLIDE 14: Conclusion (1 phút)

**Hiển thị:** Key takeaways + future work

**Nói:**

> "Kết luận: **DSP preprocessing không cải thiện accuracy** trên UrbanSound8K. p > 0.05, effect sizes negligible.
>
> Nguyên nhân: dataset sạch + feature extraction DSP ngầm → double redundancy.
>
> **Nhưng DSP không vô nghĩa**: SNR cải thiện +2.9 đến +4.5 dB, hiểu filter design giúp chọn đúng hyperparameters, và trên dataset noisy, DSP gần chắc sẽ giúp ích.
>
> **Future work**: Inject noise rồi so sánh lại, test trên real-world noisy data, data augmentation cho CNN.
>
> Cảm ơn thầy/cô."

---

## SLIDE 15: Q&A

**Hiển thị:** Technical details table

---

## Phụ lục: Câu hỏi giảng viên + Câu trả lời

### DSP (25%)

| Câu hỏi | Trả lời |
|----------|---------|
| Tại sao FIR mà không IIR? | Pha tuyến tính → bảo toàn hình dạng. IIR méo temporal shape → ảnh hưởng delta MFCC. |
| FIR order 101? | Số lẻ → đối xứng → pha tuyến tính chính xác. Group delay = 50 samples ≈ 2.27 ms. |
| filtfilt? | Lọc xuôi + ngược → phase = 0. Biên độ bị bình phương \|H\|². |
| IIR stability? | \|poles\| < 1 (trong unit circle). Butterworth order 5 → stable. |
| Spectral leakage? | Cắt cụt tín hiệu → năng lượng rò sang tần số lân cận. Hann: sidelobe -31 dB. |
| Pre-emphasis? | Thông cao bậc 1, bù phổ nghiêng -6 dB/octave, MFCC bắt tốt tần số cao. |
| Passband 50–10000? | PSD: <50 Hz = DC + rung. >10000: f_high(99.9%) = 10304 Hz. |
| Wavelet transform? | CWT: adaptive resolution, tốt cho transient (gun_shot). DWT: phân tích energy per band, xác nhận PSD. Dùng để phân tích, không làm feature. |

### AI/ML (20%)

| Câu hỏi | Trả lời |
|----------|---------|
| RF > CNN tại sao? | Dataset nhỏ (~7800/fold). CNN cần nhiều data. RF xử lý high-dim tốt. |
| Overfitting? | CNN: early stopping + dropout. RF: 500 trees. SVM: regularization C. |
| SVM cần PCA? | 931 dim + RBF → O(n²d). PCA(200) giữ ~95% variance. |
| Cross-validation? | 10-fold predefined, không shuffle. Đúng protocol UrbanSound8K. |

### Critical Discussion (10%)

| Câu hỏi | Trả lời |
|----------|---------|
| DSP không giúp — giá trị gì? | "Không giúp" cũng là phát hiện khoa học. Chứng minh khi nào redundant. Dataset sạch là root cause. |
| Inject noise thì sao? | Pipeline B sẽ thắng. Future work: inject Gaussian noise SNR 0–10 dB. |
| DSP cần cho DL? | Clean data: không. DL tự học. Noisy data: giúp hội tụ nhanh. |
| Dataset bias? | Class imbalance xử lý bằng F1-macro. Không có gender/age bias vì environmental sounds. |
| Tại sao không dùng data augmentation? | Thời gian hạn chế. Đề xuất trong future work: time-shift, pitch-shift, noise injection. |
| Real-world applicability? | Cần test trên ESC-50, AudioSet, hoặc field recordings. UrbanSound8K quá sạch cho kết luận tổng quát. |
