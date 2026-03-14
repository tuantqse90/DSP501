# Dataset — UrbanSound8K

## 1. Tổng Quan

**UrbanSound8K** là bộ dữ liệu âm thanh môi trường đô thị, được tạo bởi **Justin Salamon, Christopher Jacoby và Juan Pablo Bello** tại **NYU (New York University)** năm 2014.

| Thông tin | Giá trị |
|-----------|---------|
| Tổng số clip | **8,732** |
| Số lớp (class) | **10** |
| Số fold | **10** (chia sẵn) |
| Định dạng | WAV |
| Độ dài mỗi clip | Tối đa **4 giây** |
| Nguồn gốc | Freesound.org |
| Số bản ghi gốc | **1,297** bản ghi độc lập |

---

## 2. Các Lớp Âm Thanh (10 Classes)

Mỗi clip thuộc 1 trong 10 loại âm thanh thường gặp trong thành phố:

| classID | Tên lớp | Tiếng Việt | Số lượng clip |
|---------|---------|------------|---------------|
| 0 | air_conditioner | Máy điều hòa | 1,000 |
| 1 | car_horn | Còi xe hơi | 429 |
| 2 | children_playing | Trẻ em chơi đùa | 1,000 |
| 3 | dog_bark | Chó sủa | 1,000 |
| 4 | drilling | Khoan đường | 1,000 |
| 5 | engine_idling | Động cơ chạy không tải | 1,000 |
| 6 | gun_shot | Tiếng súng | 374 |
| 7 | jackhammer | Búa khoan | 1,000 |
| 8 | siren | Còi báo động | 929 |
| 9 | street_music | Nhạc đường phố | 1,000 |

**Nhận xét**: Dataset **không cân bằng** (imbalanced). Các lớp `car_horn` (429) và `gun_shot` (374) có ít hơn đáng kể so với các lớp khác (1,000). Điều này có thể ảnh hưởng đến hiệu suất model.

---

## 3. Cấu Trúc File CSV (Metadata)

File `UrbanSound8K.csv` nằm tại `data/UrbanSound8K/metadata/UrbanSound8K.csv`, chứa **8 cột**:

| Cột | Kiểu dữ liệu | Ý nghĩa |
|-----|---------------|---------|
| `slice_file_name` | string | Tên file audio (VD: `100032-3-0-0.wav`) |
| `fsID` | int | **Freesound ID** — mã định danh bản ghi gốc trên Freesound.org |
| `start` | float | Thời điểm bắt đầu (giây) của clip trong bản ghi gốc |
| `end` | float | Thời điểm kết thúc (giây) của clip trong bản ghi gốc |
| `salience` | int | Mức nổi bật: **1** = foreground (rõ ràng), **2** = background (nền) |
| `fold` | int | Số fold (1–10), dùng cho cross-validation |
| `classID` | int | Mã số lớp (0–9) |
| `class` | string | Tên lớp âm thanh |

### Ví dụ dữ liệu

```
slice_file_name       fsID    start    end      salience  fold  classID  class
100032-3-0-0.wav      100032  0.0      0.317    1         5     3        dog_bark
100263-2-0-117.wav    100263  58.5     62.5     1         5     2        children_playing
100263-2-0-121.wav    100263  60.5     64.5     1         5     2        children_playing
```

---

## 4. Quy Tắc Đặt Tên File

Mỗi file audio có tên theo format:

```
[fsID]-[classID]-[occurrenceID]-[sliceID].wav
```

| Thành phần | Ý nghĩa | Ví dụ |
|------------|---------|-------|
| `fsID` | ID bản ghi gốc trên Freesound | `100032` |
| `classID` | Mã lớp (0–9) | `3` = dog_bark |
| `occurrenceID` | Lần xuất hiện thứ mấy của âm thanh đó trong bản ghi | `0` = lần đầu tiên |
| `sliceID` | Đoạn cắt thứ mấy từ lần xuất hiện đó | `0` = đoạn đầu tiên |

**Ví dụ**: `180937-7-3-15.wav`
- `180937` → Bản ghi gốc ID 180937 trên Freesound
- `7` → Lớp jackhammer
- `3` → Lần xuất hiện thứ 4 (đếm từ 0)
- `15` → Đoạn cắt thứ 16

---

## 5. Fold Là Gì?

### Khái niệm

**Fold** là cách chia dataset thành **10 nhóm** để thực hiện **10-fold cross-validation**. Tác giả dataset đã chia sẵn — ta **KHÔNG ĐƯỢC** tự shuffle lại.

### Phân bố clip theo fold

| Fold | Tổng clip | Nguồn gốc (fsID) |
|------|-----------|-------------------|
| 1 | 873 | 134 |
| 2 | 888 | 134 |
| 3 | 925 | 134 |
| 4 | 990 | 134 |
| 5 | 936 | 132 |
| 6 | 823 | 130 |
| 7 | 838 | 129 |
| 8 | 806 | 126 |
| 9 | 816 | 125 |
| 10 | 837 | 124 |

### Cách sử dụng

Lặp 10 lần, mỗi lần:
- **Test**: 1 fold (VD: fold 3)
- **Train**: 9 fold còn lại (fold 1, 2, 4, 5, 6, 7, 8, 9, 10)

```
Lần 1: Train [2,3,4,5,6,7,8,9,10] → Test [1]
Lần 2: Train [1,3,4,5,6,7,8,9,10] → Test [2]
Lần 3: Train [1,2,4,5,6,7,8,9,10] → Test [3]
...
Lần 10: Train [1,2,3,4,5,6,7,8,9] → Test [10]
```

Kết quả cuối cùng = **trung bình accuracy của 10 lần**.

### Tại sao KHÔNG được tự shuffle?

Một bản ghi gốc (`fsID`) có thể tạo ra **nhiều clip**. Ví dụ:
- `fsID = 100263` tạo ra 4 clip: `100263-2-0-117.wav`, `100263-2-0-121.wav`, `100263-2-0-126.wav`, `100263-2-0-137.wav`

Nếu tự shuffle, có thể clip `100263-2-0-117.wav` rơi vào tập train và clip `100263-2-0-121.wav` rơi vào tập test. Hai clip này gần giống nhau (cùng nguồn, cách nhau chỉ 2 giây) → model "gian lận" bằng cách nhớ nguồn thay vì học đặc trưng thực sự.

Đây gọi là **data leakage** — kết quả sẽ cao ảo nhưng model thực tế kém.

Tác giả dataset đã chia sẵn fold sao cho **tất cả clip từ cùng 1 fsID nằm chung 1 fold**.

---

## 6. Salience (Mức Nổi Bật)

| Giá trị | Ý nghĩa | Số lượng |
|---------|---------|----------|
| **1** (Foreground) | Âm thanh mục tiêu **rõ ràng**, là âm thanh chính | 5,702 clip (65.3%) |
| **2** (Background) | Âm thanh mục tiêu **bị lẫn** với tiếng ồn nền | 3,030 clip (34.7%) |

Ví dụ: Một clip gán nhãn `dog_bark` với `salience=2` nghĩa là tiếng chó sủa có trong clip nhưng bị che bởi tiếng ồn khác (xe cộ, gió, v.v.).

---

## 7. Thống Kê Độ Dài Audio

| Thống kê | Giá trị |
|----------|---------|
| Trung bình | 3.61 giây |
| Độ lệch chuẩn | 0.97 giây |
| Ngắn nhất | 0.05 giây |
| Dài nhất | 4.00 giây |
| Trung vị | 4.00 giây |

Phần lớn clip đã có độ dài 4 giây. Trong project, ta **pad thêm zero** cho clip ngắn và **cắt bớt** cho clip dài → tất cả đều có **88,200 samples** (4 giây × 22,050 Hz).

---

## 8. Phân Bố Clip Theo Fold và Class

```
                air_conditioner  car_horn  children_playing  dog_bark  drilling  engine_idling  gun_shot  jackhammer  siren  street_music
Fold 1                     100        36               100       100       100             96        35         120     86           100
Fold 2                     100        42               100       100       100            100        35         120     91           100
Fold 3                     100        43               100       100       100            107        36         120    119           100
Fold 4                     100        59               100       100       100            107        38         120    166           100
Fold 5                     100        98               100       100       100            107        40         120     71           100
Fold 6                     100        28               100       100       100            107        46          68     74           100
Fold 7                     100        28               100       100       100            106        51          76     77           100
Fold 8                     100        30               100       100       100             88        30          78     80           100
Fold 9                     100        32               100       100       100             89        31          82     82           100
Fold 10                    100        33               100       100       100             93        32          96     83           100
```

**Nhận xét**:
- Các lớp `air_conditioner`, `children_playing`, `dog_bark`, `drilling`, `street_music` phân bố đều (100 clip/fold)
- Các lớp `car_horn`, `gun_shot`, `siren`, `engine_idling`, `jackhammer` phân bố không đều giữa các fold

---

## 9. Cấu Trúc Thư Mục

```
data/UrbanSound8K/
├── metadata/
│   └── UrbanSound8K.csv          ← File metadata (8,732 dòng)
└── audio/
    ├── fold1/                     ← 873 file .wav
    ├── fold2/                     ← 888 file .wav
    ├── fold3/                     ← 925 file .wav
    ├── fold4/                     ← 990 file .wav
    ├── fold5/                     ← 936 file .wav
    ├── fold6/                     ← 823 file .wav
    ├── fold7/                     ← 838 file .wav
    ├── fold8/                     ← 806 file .wav
    ├── fold9/                     ← 816 file .wav
    └── fold10/                    ← 837 file .wav
```

---

## 10. Cách Project Sử Dụng Dataset

### Bước 1: Load metadata
```python
df = pd.read_csv("data/UrbanSound8K/metadata/UrbanSound8K.csv")
# → DataFrame 8,732 dòng × 8 cột
```

### Bước 2: Load audio theo fold
```python
# Lấy tất cả file trong fold 1
X, y = load_fold_data(metadata, fold_ids=[1])
# X.shape = (873, 88200)  ← 873 clip, mỗi clip 88,200 samples
# y.shape = (873,)         ← nhãn classID (0-9)
```

### Bước 3: Xử lý audio
```python
# Resample → 22,050 Hz (nếu file gốc khác tần số)
# Pad/Truncate → 88,200 samples (4 giây)
# Pipeline A: Giữ nguyên raw audio
# Pipeline B: FIR bandpass → pre-emphasis → normalize
```

### Bước 4: Trích xuất đặc trưng
```python
# Handcrafted: MFCC + spectral features → vector 931 chiều
# CNN: Mel spectrogram → ma trận 128 × 173
```

### Bước 5: 10-fold cross-validation
```python
for test_fold in range(1, 11):
    train_folds = [f for f in range(1, 11) if f != test_fold]
    X_train, y_train = load_fold_data(metadata, train_folds)
    X_test, y_test = load_fold_data(metadata, [test_fold])
    # Train model → Evaluate → Lưu metrics
```

---

## 11. Tham Khảo

- **Paper**: J. Salamon, C. Jacoby, J.P. Bello, *"A Dataset and Taxonomy for Urban Sound Research"*, ACM Multimedia, 2014
- **Website**: https://urbansounddataset.weebly.com/urbansound8k.html
- **Nguồn audio**: https://freesound.org
