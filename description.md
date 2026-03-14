# Giải Thích Dự Án DSP501 — Phân Loại Âm Thanh Môi Trường

## Dự án này làm gì?

Dự án này trả lời một câu hỏi đơn giản:

> **Nếu mình "lọc sạch" âm thanh trước khi đưa vào AI, liệu AI có phân loại chính xác hơn không?**

Cụ thể, mình thu thập 8,732 đoạn âm thanh đô thị (tiếng còi xe, tiếng chó sủa, tiếng máy khoan...) rồi thử 2 cách:

- **Cách A**: Đưa âm thanh thô (raw) thẳng vào AI → xem AI đoán đúng bao nhiêu %
- **Cách B**: Lọc nhiễu + xử lý tín hiệu (DSP) trước → rồi mới đưa vào AI → xem có tốt hơn không

Kết quả: **Không có sự khác biệt đáng kể.** AI đoán đúng xấp xỉ nhau cho cả 2 cách.

---

## Tại sao lại làm dự án này?

Trong lĩnh vực xử lý tín hiệu số (DSP — Digital Signal Processing), một niềm tin phổ biến là:

> "Dữ liệu sạch hơn → AI học tốt hơn → Kết quả chính xác hơn"

Dự án này kiểm chứng khoa học xem điều đó có đúng không. Đây là bài final project cho môn DSP501, yêu cầu kết hợp kiến thức DSP (lọc tín hiệu, phân tích tần số) với AI/Machine Learning.

---

## Dataset: UrbanSound8K

### Dữ liệu là gì?

8,732 đoạn âm thanh ngắn (tối đa 4 giây) được thu thập từ môi trường đô thị, chia thành 10 loại:

| STT | Loại âm thanh | Tiếng Việt | Số lượng | Đặc điểm |
|-----|--------------|------------|----------|-----------|
| 1 | air_conditioner | Máy lạnh | 1,000 | Ù đều, không thay đổi |
| 2 | car_horn | Còi xe | 429 | Ngắn, đột ngột |
| 3 | children_playing | Trẻ em chơi | 1,000 | Ồn ào, nhiều giọng |
| 4 | dog_bark | Chó sủa | 1,000 | Ngắn, lặp lại |
| 5 | drilling | Máy khoan | 1,000 | Rung liên tục |
| 6 | engine_idling | Động cơ nổ | 1,000 | Ù đều |
| 7 | gun_shot | Tiếng súng | 374 | Rất ngắn, mạnh |
| 8 | jackhammer | Máy đục bê tông | 1,000 | Rung mạnh, lặp |
| 9 | siren | Còi báo động | 929 | Lên xuống đều |
| 10 | street_music | Nhạc đường phố | 1,000 | Phức tạp, nhiều nhạc cụ |

### Tại sao dùng dataset này?

- Đây là dataset chuẩn trong nghiên cứu âm thanh môi trường (được trích dẫn hơn 2,000 lần)
- Có sẵn 10 fold (nhóm) để đánh giá công bằng
- Đủ lớn để train AI nhưng không quá nặng cho laptop

---

## Quy trình thực hiện (6 bước)

### Bước 1: Phân tích tín hiệu (`01_signal_analysis.ipynb`)

**Mục đích**: Hiểu đặc điểm của từng loại âm thanh trước khi xử lý.

**Làm gì**:
- Vẽ dạng sóng (waveform) — hình dạng âm thanh theo thời gian
- Phân tích tần số (FFT) — âm thanh chứa những tần số nào?
- Mật độ phổ công suất (PSD) — năng lượng tập trung ở tần số nào?
- Kiểm tra tính dừng (stationarity) — âm thanh có ổn định hay thay đổi liên tục?

**Phát hiện quan trọng**:
- Máy lạnh, động cơ nổ: tín hiệu **dừng** (phổ tần số ổn định theo thời gian)
- Còi xe, chó sủa, tiếng súng: tín hiệu **không dừng** (phổ thay đổi liên tục)

> Hình dung đơn giản: Tiếng máy lạnh giống như nghe "ùùùùù" đều đều. Tiếng chó sủa giống "gâu! ... gâu!" — có lúc im, có lúc ồn.

### Bước 2: Thiết kế bộ lọc DSP (`02_dsp_pipeline.ipynb`)

**Mục đích**: Thiết kế bộ lọc để "làm sạch" âm thanh.

**Bộ lọc FIR (Finite Impulse Response)**:
- Loại: Bandpass (lọc dải thông) — chỉ giữ lại tần số từ 50 Hz đến 10,000 Hz
- Tại sao 50–10,000 Hz? Vì âm thanh hữu ích cho con người nằm trong khoảng này. Dưới 50 Hz là nhiễu nền (DC offset), trên 10,000 Hz là nhiễu cao tần.
- Bậc 101 (101 hệ số) — đủ sắc để lọc nhưng không quá nặng tính toán
- **Pha tuyến tính** — điều này quan trọng vì nó giữ nguyên thứ tự thời gian của âm thanh (không làm méo)

**Hình dung**: Giống như bạn đeo tai nghe chống ồn — nó loại bỏ tiếng gió, tiếng ù, chỉ giữ lại giọng nói và âm nhạc.

**Bộ lọc IIR (Infinite Impulse Response)** — để so sánh:
- Butterworth bậc 5
- Hiệu quả hơn (ít hệ số) nhưng pha **không** tuyến tính → có thể làm méo tín hiệu

**Quyết định**: Chọn FIR vì pha tuyến tính, dù cần nhiều hệ số hơn.

**Pre-emphasis (Nhấn mạnh tần số cao)**:
- Công thức: `y[n] = x[n] - 0.97 * x[n-1]`
- Tăng cường tần số cao (hữu ích cho phân biệt phụ âm trong lời nói, hoặc các âm sắc nét)

**Chuẩn hóa biên độ**:
- Chia tất cả giá trị cho giá trị lớn nhất → biên độ nằm trong [-1, 1]
- Đảm bảo âm thanh to/nhỏ khác nhau được đưa về cùng mức

### Bước 3: Trích xuất đặc trưng (`03_feature_engineering.ipynb`)

**Mục đích**: Biến âm thanh thành dãy số mà AI có thể hiểu.

AI không "nghe" được âm thanh. Mình phải chuyển mỗi đoạn âm 4 giây thành một vector (dãy số) mô tả đặc điểm của nó.

**Đặc trưng thủ công (931 chiều)** — cho SVM và Random Forest:

| Đặc trưng | Giải thích đơn giản | Số chiều |
|-----------|---------------------|----------|
| **MFCC** (40 hệ số + đạo hàm) | "Dấu vân tay" của âm thanh — mô tả hình dạng phổ tần theo cách tai người nghe | 840 |
| **Spectral centroid** | "Trọng tâm" tần số — âm thanh nghe "sáng" hay "trầm"? | 7 |
| **Spectral bandwidth** | Phổ tần rộng hay hẹp? | 7 |
| **Spectral rolloff** | Tần số mà dưới đó chứa 85% năng lượng | 7 |
| **Spectral flatness** | Âm thanh giống tiếng ồn trắng hay có cao độ rõ? | 7 |
| **Zero-crossing rate** | Tín hiệu đổi dấu bao nhiêu lần? (cao = nhiều nhiễu/tần số cao) | 7 |
| **RMS energy** | Âm lượng trung bình | 7 |
| **Spectral contrast** | Sự khác biệt giữa đỉnh và đáy trong phổ (7 dải tần) | 49 |
| **Tổng cộng** | | **931** |

Mỗi đặc trưng được tính qua 7 thống kê: trung bình, độ lệch chuẩn, min, max, trung vị, độ lệch, độ nhọn.

**Mel Spectrogram (128 × 173)** — cho CNN:
- Hình ảnh 2D biểu diễn năng lượng âm thanh theo thời gian và tần số
- 128 dải tần mel × 173 khung thời gian
- CNN "nhìn" bức ảnh này giống như nhìn ảnh chụp → nhận diện pattern

> Hình dung: MFCC giống như bạn mô tả giọng nói bằng lời ("giọng trầm, nói nhanh, hơi khàn"). Mel spectrogram giống như chụp ảnh sóng âm.

### Bước 4: Pipeline A — Baseline (`04_pipeline_a_raw.ipynb`)

**Mục đích**: Đưa âm thanh thô (KHÔNG lọc) vào AI, xem kết quả.

Quy trình: Âm thanh thô → Trích xuất đặc trưng → Train AI → Đánh giá

3 mô hình AI:

1. **SVM (Support Vector Machine)**
   - Tìm siêu phẳng (đường ranh giới) tối ưu để tách 10 loại âm thanh
   - Dùng kernel RBF (hàm cơ sở xuyên tâm) — có thể tách dữ liệu phi tuyến
   - Cần giảm chiều từ 931 → 200 (PCA) vì SVM chậm với nhiều chiều

2. **Random Forest (Rừng ngẫu nhiên)**
   - Xây 500 cây quyết định, mỗi cây "vote" cho 1 lớp
   - Kết quả cuối = lớp được vote nhiều nhất
   - Nhanh, ổn định, xử lý được 931 chiều trực tiếp

3. **CNN-2D (Mạng nơ-ron tích chập)**
   - Nhận input là mel spectrogram (ảnh 128×173)
   - 4 lớp convolution: học các pattern cục bộ (cạnh, kết cấu, hình dạng)
   - Cuối cùng: lớp fully connected phân loại thành 10 lớp
   - Train trên GPU (Apple MPS)

### Bước 5: Pipeline B — Có DSP (`05_pipeline_b_dsp.ipynb`)

**Mục đích**: Lọc âm thanh trước → rồi mới đưa vào AI.

Quy trình: Âm thanh thô → **Lọc FIR** → **Pre-emphasis** → **Chuẩn hóa** → Trích xuất đặc trưng → Train AI → Đánh giá

Dùng cùng 3 mô hình AI, cùng cách đánh giá — chỉ khác là dữ liệu đã được "làm sạch".

### Bước 6: So sánh thống kê (`06_comparative_analysis.ipynb`)

**Mục đích**: So sánh Pipeline A vs B một cách khoa học.

Không chỉ nhìn con số rồi nói "cái này cao hơn" — mình dùng kiểm định thống kê:

- **Paired t-test**: Kiểm tra xem sự khác biệt có ý nghĩa thống kê không (p < 0.05?)
- **Wilcoxon test**: Phiên bản phi tham số (không giả định phân phối chuẩn)
- **Cohen's d**: Đo kích thước hiệu ứng — sự khác biệt có đáng kể trong thực tế không?

---

## Cách đánh giá: 10-Fold Cross-Validation

Dataset được chia sẵn thành 10 nhóm (fold). Mình lần lượt:

```
Lần 1: Train trên fold 2-10, test trên fold 1
Lần 2: Train trên fold 1,3-10, test trên fold 2
Lần 3: Train trên fold 1-2,4-10, test trên fold 3
...
Lần 10: Train trên fold 1-9, test trên fold 10
```

→ Mỗi đoạn âm thanh được test đúng 1 lần. Kết quả cuối = trung bình 10 lần ± khoảng tin cậy 95%.

**Tại sao không chia random?** Vì UrbanSound8K có nhiều clip từ cùng 1 nguồn gốc. Nếu chia random, clip train và test có thể đến từ cùng 1 bản ghi → kết quả bị "thổi phồng" (data leakage). Fold sẵn đảm bảo không bị vấn đề này.

---

## Kết quả

### Bảng tổng hợp

| Mô hình | Pipeline A (Thô) | Pipeline B (Đã lọc) | Chênh lệch | p-value | Có ý nghĩa? |
|---------|:-----------------:|:-------------------:|:-----------:|:-------:|:----------:|
| SVM | 70.1% | 70.0% | −0.12% | 0.846 | **Không** |
| Random Forest | **71.5%** | 71.2% | −0.26% | 0.768 | **Không** |
| CNN-2D | 66.7% | 67.6% | +0.94% | 0.625 | **Không** |

### Giải thích kết quả

- **p-value > 0.05** cho cả 3 mô hình → sự khác biệt **KHÔNG** có ý nghĩa thống kê
- **Cohen's d < 0.2** → kích thước hiệu ứng **không đáng kể**
- Nói đơn giản: Pipeline A và Pipeline B cho kết quả **như nhau**

### Tại sao lọc tín hiệu không giúp ích?

Đây là phát hiện quan trọng nhất. Lý do:

1. **MFCC đã tự lọc rồi**: Khi tính MFCC, bước đầu tiên là áp dụng mel filterbank — bản chất là một loạt bộ lọc bandpass. Vậy dù mình có lọc hay không, MFCC vẫn chỉ "nhìn thấy" dải tần hữu ích.

2. **Mel spectrogram đã giới hạn tần số**: Mình đặt $f_{min}=50$ Hz, $f_{max}=10,000$ Hz — đúng bằng dải thông của bộ lọc FIR. Vậy mel spectrogram đã tự loại bỏ tần số ngoài dải.

3. **StandardScaler đã chuẩn hóa**: Trước khi đưa vào SVM/RF, features được chuẩn hóa (trung bình=0, phương sai=1). Hiệu ứng tương tự như amplitude normalization.

4. **Thống kê tổng hợp đã khử nhiễu**: Khi tính trung bình, độ lệch chuẩn... qua toàn bộ các frame, nhiễu ngẫu nhiên đã bị triệt tiêu.

> **Tóm lại**: Quá trình trích xuất đặc trưng hiện đại (MFCC, mel spectrogram) đã "ngầm" thực hiện những gì DSP preprocessing làm. Lọc thêm = lọc 2 lần → không thêm lợi ích.

---

## Cấu trúc dự án

```
DSP501/
├── config.py                    # Tất cả tham số (tần số, bậc lọc, hyperparameters...)
├── report.md                    # Báo cáo kỹ thuật đầy đủ
├── presentation.md              # Slide trình bày (15 slides)
├── progress.md                  # Nhật ký tiến độ
│
├── src/                         # Code chính
│   ├── data_loader.py           # Đọc dataset, chia fold
│   ├── signal_analysis.py       # Phân tích tín hiệu (FFT, PSD, stationarity)
│   ├── dsp_pipeline.py          # Thiết kế bộ lọc FIR/IIR, pre-emphasis
│   ├── feature_extraction.py    # Trích xuất MFCC, spectral features, mel spectrogram
│   ├── evaluation.py            # Tính metrics, t-test, Cohen's d
│   ├── visualization.py         # Vẽ tất cả biểu đồ (dark theme)
│   └── models/
│       ├── classical_ml.py      # SVM, Random Forest
│       └── deep_learning.py     # CNN-2D, CNN-1D (PyTorch)
│
├── notebooks/                   # 6 notebook chạy tuần tự
│   ├── 01_signal_analysis.ipynb
│   ├── 02_dsp_pipeline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_pipeline_a_raw.ipynb
│   ├── 05_pipeline_b_dsp.ipynb
│   └── 06_comparative_analysis.ipynb
│
├── results/
│   ├── figures/                 # 42 biểu đồ (waveform, FFT, filter, confusion matrix...)
│   └── tables/                  # Bảng kết quả CSV
│
├── data/UrbanSound8K/           # Dataset (không push lên GitHub)
└── docs/                        # Tài liệu kỹ thuật (architecture, model spec...)
```

---

## Thuật ngữ chính

| Thuật ngữ | Tiếng Việt | Giải thích |
|-----------|-----------|------------|
| DSP | Xử lý tín hiệu số | Dùng thuật toán để lọc, biến đổi tín hiệu |
| FIR Filter | Bộ lọc đáp ứng xung hữu hạn | Bộ lọc mà output chỉ phụ thuộc vào input (không hồi tiếp) |
| IIR Filter | Bộ lọc đáp ứng xung vô hạn | Bộ lọc có hồi tiếp (output phụ thuộc vào cả output trước đó) |
| Bandpass | Lọc dải thông | Chỉ cho qua tần số trong một khoảng nhất định |
| Pre-emphasis | Nhấn mạnh tần số cao | Bộ lọc high-pass đơn giản, tăng cường thành phần tần số cao |
| FFT | Biến đổi Fourier nhanh | Phân tích tín hiệu thành các thành phần tần số |
| MFCC | Hệ số cepstral tần số mel | "Dấu vân tay" của âm thanh, mô phỏng cách tai người nghe |
| Mel Spectrogram | Phổ tần mel | Ảnh 2D biểu diễn năng lượng theo thời gian × tần số (thang mel) |
| SVM | Máy vector hỗ trợ | Thuật toán ML tìm siêu phẳng tối ưu để phân tách dữ liệu |
| Random Forest | Rừng ngẫu nhiên | Tập hợp nhiều cây quyết định, vote để phân loại |
| CNN | Mạng nơ-ron tích chập | Mạng deep learning chuyên xử lý dữ liệu dạng lưới (ảnh, spectrogram) |
| Cross-validation | Kiểm chứng chéo | Chia dữ liệu thành nhiều phần, luân phiên train/test |
| p-value | Giá trị p | Xác suất quan sát được kết quả này nếu không có sự khác biệt thật |
| Cohen's d | Kích thước hiệu ứng Cohen | Đo mức độ khác biệt thực tế (không chỉ thống kê) |
| Paired t-test | Kiểm định t ghép cặp | So sánh 2 nhóm kết quả có cùng cặp (cùng fold) |

---

## Kết luận cuối cùng

1. **DSP preprocessing KHÔNG cải thiện phân loại âm thanh** khi kết hợp với feature extraction hiện đại
2. **Random Forest** là mô hình tốt nhất (71.5%) — đơn giản, nhanh, ổn định
3. **CNN thua classical ML** trên dataset nhỏ (~8,700 mẫu) — deep learning cần nhiều dữ liệu hơn
4. Dù không cải thiện accuracy, việc hiểu DSP vẫn quan trọng — nó giúp hiểu tại sao feature extraction hoạt động, và trong những bài toán khác (speech enhancement, noise cancellation) thì DSP vẫn thiết yếu
