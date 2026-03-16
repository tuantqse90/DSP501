# DSP501 — Script Thuyết Trình (60 phút, 5 người)

## Tổng quan phân công

| # | Người | Phần | Thời gian | Slides |
|---|--------|------|-----------|--------|
| 1 | **Tuấn** | Mở đầu + Bài toán + Dataset | 10 phút | 1-4 |
| 2 | **Hải** | DSP Theory + Pipeline B | 12 phút | 5-9 |
| 3 | **Phú** | Feature Extraction + Models | 12 phút | 10-14 |
| 4 | **Vĩnh** | Thí nghiệm + Kết quả + Phân tích | 12 phút | 15-20 |
| 5 | **Tuấn Bự** | Demo live + Kết luận + Q&A | 14 phút | 21-24 |

---

## PHẦN 1 — Tuấn (10 phút)
### "Giới thiệu + Bài toán + Dataset"

**[Slide 1 — Title] (1 phút)**
> Xin chào mọi người. Hôm nay nhóm mình sẽ trình bày đề tài **"Environmental Sound Classification — So sánh Pipeline có và không có DSP preprocessing"**.
>
> Nhóm gồm 5 thành viên: Tuấn, Hải, Phú, Vĩnh và Tuấn Bự.

**[Slide 2 — Bài toán] (3 phút)**
> Bài toán của mình là **phân loại âm thanh môi trường** — ví dụ tiếng còi xe, tiếng chó sủa, tiếng máy khoan...
>
> Ứng dụng thực tế rất nhiều: giám sát tiếng ồn đô thị, hỗ trợ người khiếm thính, smart city, an ninh...
>
> **Câu hỏi nghiên cứu chính**: Nếu mình thêm bước tiền xử lý DSP (lọc nhiễu, pre-emphasis, normalize) trước khi đưa vào AI, thì kết quả có tốt hơn không?
>
> Mình chia thành 2 pipeline:
> - **Pipeline A**: Audio thô → trích đặc trưng → AI
> - **Pipeline B**: Audio thô → DSP xử lý → trích đặc trưng → AI

**[Slide 3 — Dataset UrbanSound8K] (4 phút)**
> Dataset mình dùng là **UrbanSound8K** — một benchmark phổ biến cho bài toán này.
>
> - **8732 audio clips**, mỗi clip ≤ 4 giây
> - **10 classes**: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
> - Chia sẵn **10 folds** — rất quan trọng: mình KHÔNG ĐƯỢC shuffle data qua các fold vì cùng một nguồn âm thanh có thể nằm ở nhiều clip
>
> *(Cho nghe 2-3 mẫu audio nhanh)*
>
> Mỗi class có phân bố không đều — ví dụ air_conditioner và jackhammer có nhiều clip dài, còn gun_shot thì rất ngắn. Đây là thách thức thực tế.

**[Slide 4 — Tổng quan phương pháp] (2 phút)**
> Đây là sơ đồ tổng quan 2 pipeline. Mình sẽ so sánh 3 model: **SVM, Random Forest, và CNN-2D** trên cả 2 pipeline.
>
> Đánh giá bằng **10-fold cross-validation** theo fold có sẵn, dùng **paired t-test** và **Cohen's d** để kiểm định thống kê.
>
> Bây giờ mình mời Hải lên trình bày phần DSP.

---

## PHẦN 2 — Hải (12 phút)
### "Lý thuyết DSP + Thiết kế Pipeline B"

**[Slide 5 — Tại sao cần DSP?] (2 phút)**
> Audio trong thực tế thường bị nhiễu — tiếng gió, tiếng nền, DC offset...
>
> Ý tưởng của Pipeline B là: **trước khi trích đặc trưng, mình "làm sạch" tín hiệu** bằng các kỹ thuật DSP kinh điển. Giả thuyết là model sẽ học tốt hơn trên tín hiệu sạch.

**[Slide 6 — FIR Bandpass Filter] (4 phút)**
> Bước 1: **Lọc thông dải FIR** (50 Hz – 10 kHz).
>
> - Tần số dưới 50 Hz chủ yếu là DC offset và nhiễu rung — không hữu ích cho phân loại
> - Trên 10 kHz thường là nhiễu cao tần
> - Mình dùng **FIR order 101, cửa sổ Hann**
> - FIR có ưu điểm: **linear phase** — không làm méo pha tín hiệu, quan trọng khi trích đặc trưng thời gian
>
> *(Show đáp ứng tần số của filter)*
>
> Dùng `scipy.signal.firwin` với `filtfilt` để lọc zero-phase.

**[Slide 7 — Pre-emphasis] (3 phút)**
> Bước 2: **Pre-emphasis** với hệ số 0.97.
>
> Công thức: `y[n] = x[n] - 0.97 * x[n-1]`
>
> Tác dụng: **tăng cường tần số cao** — vì âm thanh tự nhiên có năng lượng giảm dần theo tần số (spectral tilt). Pre-emphasis cân bằng lại phổ, giúp MFCC trích xuất tốt hơn.
>
> Kỹ thuật này rất phổ biến trong speech processing.

**[Slide 8 — Peak Normalization] (1 phút)**
> Bước 3: **Peak normalize** — chia cho giá trị tuyệt đối lớn nhất.
>
> Đưa biên độ về [-1, 1], đảm bảo các clip có cùng mức âm lượng, tránh model bị bias bởi volume.

**[Slide 9 — So sánh trực quan A vs B] (2 phút)**
> Đây là so sánh waveform và spectrogram trước/sau DSP.
>
> *(Show figure Pipeline A vs B)*
>
> Nhìn spectrogram thấy rõ: Pipeline B cắt bớt tần số thấp và cao, phổ "sáng" hơn ở vùng mid-frequency do pre-emphasis.
>
> Câu hỏi là: liệu sự khác biệt này có giúp model classify tốt hơn? Phú sẽ trình bày tiếp phần models.

---

## PHẦN 3 — Phú (12 phút)
### "Feature Extraction + Models"

**[Slide 10 — Feature Extraction] (4 phút)**
> Mình trích **931 features** cho mỗi audio clip, gồm:
>
> - **MFCC** (40 coefficients) + Delta + Delta-Delta = 120 channels, mỗi channel lấy 7 statistics (mean, std, min, max, median, skew, kurtosis) → **840 features**
> - **Spectral features**: centroid, bandwidth, rolloff, flatness → **28 features**
> - **ZCR + RMS** → **14 features**
> - **Spectral contrast** (7 bands) → **49 features**
>
> Tổng: **931 features** — đây là feature set khá comprehensive cho audio classification.
>
> Với CNN thì dùng **Mel spectrogram** (128 mel bands) trực tiếp làm input — giống như "ảnh" của âm thanh.

**[Slide 11 — SVM] (3 phút)**
> Model 1: **Support Vector Machine** với RBF kernel.
>
> - 931 features khá nhiều chiều → mình dùng **StandardScaler + PCA(200)** để giảm chiều trước
> - Hyperparameters: C ∈ {0.1, 1, 10, 100}, gamma ∈ {scale, auto, 0.01, 0.001}
> - SVM tìm hyperplane tối ưu phân tách các class trong không gian cao chiều
>
> SVM phù hợp khi feature space lớn và data không quá nhiều.

**[Slide 12 — Random Forest] (3 phút)**
> Model 2: **Random Forest** — ensemble của nhiều Decision Trees.
>
> - N_estimators ∈ {100, 200, 500}, max_depth ∈ {10, 20, 50, None}
> - Ưu điểm: **không cần scale features**, ít overfit, có feature importance
> - Train rất nhanh (vài giây/fold) so với SVM và CNN
>
> Random Forest cũng cho mình biết feature nào quan trọng nhất — MFCC thường dominate.

**[Slide 13 — CNN-2D] (2 phút)**
> Model 3: **CNN-2D** — Convolutional Neural Network.
>
> - Input: Mel spectrogram (1 × 128 × T)
> - Architecture: 4 Conv blocks (32→64→128→256 filters) + Global Average Pooling + FC
> - Batch norm + Dropout 0.3, optimizer Adam lr=0.001
> - Early stopping patience=10, max 100 epochs
>
> CNN "nhìn" spectrogram như ảnh, tự học các pattern tần số-thời gian.

**[Slide 14 — Evaluation Strategy] (2 phút — chuyển tiếp cho Vĩnh)**
> Đánh giá: **10-fold CV** theo fold có sẵn của UrbanSound8K.
>
> Mỗi fold, train trên 9 fold còn lại, test trên 1 fold. Lặp lại cho cả Pipeline A và B.
>
> So sánh bằng:
> - **Paired t-test**: p-value < 0.05 thì khác biệt có ý nghĩa
> - **Cohen's d**: đo effect size (nhỏ/trung bình/lớn)
>
> Vĩnh sẽ trình bày kết quả chi tiết.

---

## PHẦN 4 — Vĩnh (12 phút)
### "Thí nghiệm + Kết quả + Phân tích"

**[Slide 15 — Tổng quan kết quả] (2 phút)**
> Đây là bảng kết quả tổng hợp 10-fold CV:
>
> | Model | Pipeline A | Pipeline B | p-value | Significant? |
> |-------|-----------|-----------|---------|--------------|
> | SVM | 70.1% | 70.0% | 0.846 | Không |
> | RF | 71.5% | 71.2% | 0.768 | Không |
> | CNN-2D | 66.7% | 67.6% | 0.625 | Không |
>
> **Kết quả bất ngờ**: không có sự khác biệt có ý nghĩa thống kê giữa 2 pipeline ở cả 3 models!

**[Slide 16 — Phân tích SVM] (2 phút)**
> SVM: Pipeline A = 70.1%, Pipeline B = 70.0%.
>
> Gần như giống hệt nhau. p-value = 0.846 — rất cao, nghĩa là sự khác biệt hoàn toàn do ngẫu nhiên.
>
> *(Show boxplot 10 folds)*

**[Slide 17 — Phân tích RF] (2 phút)**
> Random Forest đạt kết quả cao nhất: **71.5%** (Pipeline A).
>
> Pipeline B thấp hơn một chút (71.2%) nhưng p = 0.768. Cohen's d rất nhỏ.
>
> RF cũng là model train nhanh nhất — chỉ vài giây/fold.

**[Slide 18 — Phân tích CNN] (2 phút)**
> CNN-2D có accuracy thấp nhất: 66.7% (A) vs 67.6% (B).
>
> Thú vị: CNN trên Pipeline B hơi cao hơn A, nhưng vẫn không significant (p = 0.625).
>
> CNN underperform so với classical ML — có thể do dataset size nhỏ (8732 samples) chưa đủ để CNN phát huy.

**[Slide 19 — Confusion Matrix + Per-class] (2 phút)**
> *(Show confusion matrix)*
>
> Các class dễ nhận: **gun_shot** (ngắn, đặc trưng), **car_horn**, **siren**
>
> Các class hay bị nhầm: **engine_idling ↔ air_conditioner** (cùng là tiếng ồn liên tục), **drilling ↔ jackhammer** (cùng cơ khí)
>
> Pipeline B không giúp cải thiện các cặp dễ nhầm này.

**[Slide 20 — Tại sao DSP không giúp?] (2 phút)**
> 3 lý do chính:
>
> 1. **MFCC đã robust**: MFCC bản chất đã lọc và nén phổ — pre-emphasis + bandpass "trùng" với những gì MFCC tự làm
> 2. **Bandpass quá conservative**: 50 Hz–10 kHz giữ lại hầu hết thông tin — không cắt đủ nhiễu để tạo khác biệt
> 3. **Nhiễu là thông tin**: trong environmental sound, "nhiễu nền" có thể là feature hữu ích — air_conditioner chính là "tiếng ồn"!
>
> Đây là insight quan trọng: **DSP preprocessing không phải lúc nào cũng tốt** — phụ thuộc vào bài toán cụ thể.

---

## PHẦN 5 — Tuấn Bự (14 phút)
### "Demo Live + Kết luận + Q&A"

**[Slide 21 — Demo intro] (1 phút)**
> Bây giờ đến phần demo live! Mình đã build một web app bằng Gradio để mọi người trải nghiệm trực tiếp.
>
> *(Mở browser → localhost:7860)*

**[Demo — Classifier] (8 phút)**
> Web app cho phép:
> - Upload 1 file audio hoặc **thu âm trực tiếp** bằng microphone
> - Chọn model: Random Forest, SVM, hoặc CNN-2D
> - Xem kết quả **Pipeline A vs Pipeline B side-by-side**: waveform, spectrogram, và confidence bars
>
> *(Demo 1: Upload file tiếng chó sủa → chạy RF → show kết quả 2 pipeline)*
>
> Mọi người thấy không — cả 2 pipeline đều predict đúng "dog_bark", confidence gần như giống nhau.
>
> *(Demo 2: Thử thu âm trực tiếp hoặc upload tiếng khó hơn — ví dụ drilling vs jackhammer)*
>
> Đây là cặp class hay bị nhầm. Thử switch sang SVM, CNN xem model nào xử lý tốt hơn.
>
> *(Demo 3: Upload air_conditioner → switch qua 3 models)*
>
> Kết quả 2 pipeline luôn rất gần nhau — đúng như thí nghiệm Vĩnh vừa trình bày. DSP preprocessing không tạo ra khác biệt đáng kể.

**[Slide 22 — Kết luận] (3 phút)**
> Tóm lại:
>
> 1. **DSP preprocessing (FIR + pre-emphasis + normalize) KHÔNG cải thiện** accuracy đáng kể cho environmental sound classification trên UrbanSound8K
> 2. **Random Forest** là model tốt nhất (71.5%), classical ML outperform CNN trên dataset nhỏ
> 3. **MFCC features đã đủ robust** — DSP thêm vào bị "trùng" tác dụng
> 4. Bài học: **không phải lúc nào nhiều bước xử lý hơn = tốt hơn** — cần thực nghiệm để chứng minh

**[Slide 23 — Hạn chế + Hướng phát triển] (2 phút)**
> Hạn chế:
> - Chỉ test 1 cấu hình DSP (có thể filter khác sẽ cho kết quả khác)
> - CNN chưa dùng augmentation hay pretrained model
>
> Hướng phát triển:
> - Thử adaptive filtering theo từng class
> - Dùng pretrained audio models (PANNs, AST)
> - Real-time classification cho ứng dụng thực tế

**[Slide 24 — Q&A]**
> Cảm ơn mọi người đã lắng nghe! Mời thầy/cô và các bạn đặt câu hỏi.

---

## Timeline tổng hợp

```
00:00 ─── Tuấn ──────── Intro + Dataset              (10 min)
10:00 ─── Hải ────────── DSP Theory + Pipeline B      (12 min)
22:00 ─── Phú ────────── Features + Models            (12 min)
34:00 ─── Vĩnh ───────── Kết quả + Phân tích          (12 min)
46:00 ─── Tuấn Bự ────── Demo + Kết luận + Q&A        (14 min)
60:00 ─── HẾT
```

## Tips

- **Chuyển tiếp**: mỗi người khi kết thúc, giới thiệu người tiếp theo 1 câu
- **Demo**: Tuấn Bự nên test app trước, chuẩn bị sẵn 3 file audio (dog_bark, drilling, air_conditioner)
- **Q&A**: ai được hỏi về phần nào thì người đó trả lời
- **Backup**: nếu demo lỗi thì show screenshots đã chụp sẵn
