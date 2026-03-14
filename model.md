# Giải Thích Các Model & Thư Viện Scikit-Learn Trong DSP501

## Mục lục

1. [Tổng quan 3 model](#1-tổng-quan-3-model)
2. [SVM — Support Vector Machine](#2-svm--support-vector-machine)
3. [Random Forest — Rừng Ngẫu Nhiên](#3-random-forest--rừng-ngẫu-nhiên)
4. [CNN-2D — Mạng Nơ-ron Tích Chập](#4-cnn-2d--mạng-nơ-ron-tích-chập)
5. [Scikit-Learn trong dự án](#5-scikit-learn-trong-dự-án)
6. [So sánh 3 model](#6-so-sánh-3-model)

---

## 1. Tổng quan 3 model

Dự án sử dụng 3 model AI để phân loại 10 loại âm thanh môi trường:

```
Âm thanh (4 giây)
    │
    ├──→ [Trích xuất 931 features] ──→ SVM ──→ Dự đoán lớp
    │                                    │
    │                                    └──→ Random Forest ──→ Dự đoán lớp
    │
    └──→ [Mel Spectrogram 128×173] ──→ CNN-2D ──→ Dự đoán lớp
```

- **SVM** và **Random Forest**: Nhận vector 931 số (đặc trưng thủ công) → dùng thư viện **scikit-learn**
- **CNN-2D**: Nhận ảnh spectrogram 128×173 pixel → dùng thư viện **PyTorch**

---

## 2. SVM — Support Vector Machine

### 2.1 SVM là gì?

**Support Vector Machine** (Máy Vector Hỗ Trợ) là thuật toán tìm **đường ranh giới tối ưu** để phân tách các lớp dữ liệu.

**Ví dụ đơn giản**: Tưởng tượng bạn có 2 nhóm điểm trên mặt phẳng (đỏ và xanh). SVM tìm đường thẳng sao cho khoảng cách từ đường đến điểm gần nhất của mỗi nhóm là **lớn nhất** (margin tối đa).

```
    ○ ○                         ○ ○
  ○ ○ ○         SVM          ○ ○ ○    |
    ○           ──→            ○      |  ← đường ranh giới (hyperplane)
                              ────────|────────
  ● ● ●                    ● ● ●     |
    ● ●                      ● ●
      ●                        ●
```

Những điểm gần đường ranh giới nhất gọi là **Support Vectors** — chúng "chống đỡ" (support) đường ranh giới.

### 2.2 Kernel RBF — Xử lý dữ liệu phi tuyến

Trong thực tế, dữ liệu hiếm khi tách được bằng đường thẳng. **Kernel trick** biến đổi dữ liệu lên không gian cao chiều hơn, nơi chúng có thể tách tuyến tính.

**RBF (Radial Basis Function)** kernel:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

- Đo **độ tương đồng** giữa 2 điểm dữ liệu
- 2 điểm gần nhau → K ≈ 1 (giống nhau)
- 2 điểm xa nhau → K ≈ 0 (khác nhau)

**Ví dụ**: Nếu feature vector của "tiếng chó sủa" và "tiếng còi xe" rất khác nhau (khoảng cách lớn), RBF kernel sẽ cho giá trị gần 0 → SVM biết chúng thuộc lớp khác nhau.

### 2.3 Tham số quan trọng

**C (Regularization)**:
- C nhỏ (0.1): Đường ranh giới **mềm** — cho phép một số điểm bị phân loại sai → tổng quát hóa tốt hơn
- C lớn (100): Đường ranh giới **cứng** — cố gắng phân loại đúng hết → dễ overfit

```
C nhỏ (mềm):              C lớn (cứng):
  ○ ○                        ○ ○
○ ○ ● ○    (chấp nhận       ○ ○  ○   (đường cong phức tạp
──────────  1 điểm sai)     ──/──\──  để đúng hết)
● ● ● ●                    ● ● ● ●
```

**gamma (Phạm vi ảnh hưởng)**:
- gamma nhỏ: Mỗi điểm ảnh hưởng **rộng** → đường ranh giới mượt
- gamma lớn: Mỗi điểm ảnh hưởng **hẹp** → đường ranh giới phức tạp

**Trong dự án**: C=10, gamma='scale' (tự tính dựa trên variance của features)

### 2.4 SVM trong dự án (code)

File: `src/models/classical_ml.py`

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Bước 1: Chuẩn hóa features (trung bình=0, phương sai=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # (7800, 931)
X_test_scaled = scaler.transform(X_test)

# Bước 2: Giảm chiều từ 931 → 200 (SVM chậm với nhiều chiều)
pca = PCA(n_components=200)
X_train_pca = pca.fit_transform(X_train_scaled)  # (7800, 200)
X_test_pca = pca.transform(X_test_scaled)

# Bước 3: Train SVM
svm = SVC(C=10, gamma='scale', kernel='rbf')
svm.fit(X_train_pca, y_train)

# Bước 4: Dự đoán
predictions = svm.predict(X_test_pca)
```

### 2.5 Tại sao cần PCA?

SVM với RBF kernel có độ phức tạp tính toán **O(n² × d)** với n = số mẫu, d = số chiều.

- Không PCA: n=7800, d=931 → rất chậm (~9 phút/fold)
- Có PCA: n=7800, d=200 → nhanh hơn nhiều (~40s/fold)

PCA (Principal Component Analysis) giảm 931 chiều xuống 200 chiều bằng cách giữ lại những "hướng" chứa nhiều thông tin nhất, loại bỏ chiều dư thừa/nhiễu.

### 2.6 Ưu & nhược điểm SVM

| Ưu điểm | Nhược điểm |
|---------|-----------|
| Hiệu quả với dữ liệu cao chiều | Chậm khi n lớn (O(n²)) |
| Kernel trick xử lý phi tuyến | Cần chuẩn hóa features |
| Ít overfit nhờ margin tối đa | Không cho xác suất trực tiếp |
| Tốt khi số chiều > số mẫu | Nhạy cảm với C và gamma |

---

## 3. Random Forest — Rừng Ngẫu Nhiên

### 3.1 Cây quyết định (Decision Tree) là gì?

Trước khi hiểu Random Forest, cần hiểu **cây quyết định**. Nó hoạt động giống cách con người ra quyết định:

```
                    [Spectral centroid > 3000 Hz?]
                    /                              \
                 Có                                Không
                 /                                    \
    [ZCR > 0.1?]                          [MFCC_1 > -5?]
    /          \                           /            \
  Có          Không                      Có            Không
  /              \                        /                \
gun_shot     car_horn              engine_idling      air_conditioner
```

Mỗi nút (node) hỏi 1 câu hỏi về 1 feature → rẽ trái/phải → đến lá (leaf) = dự đoán.

### 3.2 Random Forest = Nhiều cây → Vote

**Vấn đề**: 1 cây quyết định dễ **overfit** (học thuộc dữ liệu train, dự đoán tệ trên dữ liệu mới).

**Giải pháp**: Tạo **500 cây** khác nhau, mỗi cây "vote" → kết quả = lớp được vote nhiều nhất.

```
Cây 1: "dog_bark"     ─┐
Cây 2: "dog_bark"      │
Cây 3: "car_horn"      ├──→ Vote: dog_bark (3/5) ──→ Kết quả: dog_bark
Cây 4: "dog_bark"      │
Cây 5: "children_playing"─┘
```

### 3.3 Tại sao "Random"?

Mỗi cây được tạo khác nhau bằng 2 cách:

1. **Bagging (Bootstrap Aggregating)**: Mỗi cây train trên một **mẫu ngẫu nhiên** (có hoàn lại) của dữ liệu. Cây 1 có thể thấy mẫu A,B,C,A,D. Cây 2 thấy B,C,C,E,A.

2. **Feature randomness**: Tại mỗi nút, cây chỉ được chọn từ một **tập con ngẫu nhiên** của features (không phải toàn bộ 931). Ví dụ: nút này chỉ được xét 30 features ngẫu nhiên.

→ 500 cây khác nhau → mỗi cây sai ở chỗ khác → khi kết hợp, lỗi triệt tiêu lẫn nhau.

### 3.4 Feature Importance

Một lợi thế lớn của Random Forest: nó cho biết **feature nào quan trọng nhất**.

Cách tính: Nếu 1 feature được dùng ở nhiều nút gần gốc (root) và giảm nhiều "tạp" (impurity) → feature đó quan trọng.

Trong dự án, kết quả feature importance cho thấy **MFCC** là nhóm feature quan trọng nhất → hợp lý vì MFCC là "dấu vân tay" của âm thanh.

### 3.5 Tham số quan trọng

**n_estimators = 500** (số cây):
- Nhiều cây hơn → kết quả ổn định hơn → nhưng chậm hơn
- 500 là đủ tốt cho bài toán này

**max_depth = None** (độ sâu tối đa):
- None = cây phát triển tối đa → mỗi cây "mạnh" nhưng dễ overfit
- Không sao vì khi kết hợp 500 cây, overfit bị triệt tiêu

**n_jobs = -1**:
- Dùng tất cả CPU cores để train song song
- 500 cây chia đều cho 8 cores → nhanh gấp ~8 lần

### 3.6 Random Forest trong dự án (code)

File: `src/models/classical_ml.py`

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Bước 1: Chuẩn hóa (không bắt buộc cho RF, nhưng để consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # (7800, 931)
X_test_scaled = scaler.transform(X_test)

# Bước 2: Train Random Forest (KHÔNG cần PCA — RF xử lý 931 chiều tốt)
rf = RandomForestClassifier(
    n_estimators=500,      # 500 cây
    max_depth=None,        # Không giới hạn độ sâu
    random_state=42,       # Seed để tái tạo kết quả
    n_jobs=-1              # Dùng tất cả CPU cores
)
rf.fit(X_train_scaled, y_train)

# Bước 3: Dự đoán (500 cây vote)
predictions = rf.predict(X_test_scaled)

# Bonus: Xem feature nào quan trọng
importances = rf.feature_importances_  # array (931,)
```

### 3.7 Ưu & nhược điểm Random Forest

| Ưu điểm | Nhược điểm |
|---------|-----------|
| Không cần chuẩn hóa features | Không giải thích được từng dự đoán |
| Xử lý tốt dữ liệu cao chiều | Dùng nhiều RAM (500 cây) |
| Cho biết feature importance | Chậm hơn 1 cây đơn lẻ |
| Ít overfit nhờ bagging | Không tốt bằng deep learning trên big data |
| Nhanh nhờ parallel training | |

---

## 4. CNN-2D — Mạng Nơ-ron Tích Chập

### 4.1 CNN là gì?

**Convolutional Neural Network** (Mạng Nơ-ron Tích Chập) là kiến trúc deep learning chuyên xử lý dữ liệu dạng **lưới** (ảnh, spectrogram). Thay vì đưa toàn bộ ảnh vào neural network, CNN dùng **bộ lọc nhỏ** (kernel) trượt qua ảnh để phát hiện pattern cục bộ.

### 4.2 Ý tưởng cốt lõi

```
Input: Mel Spectrogram (128×173 pixel)
  │
  │  ┌─────┐
  │  │ 3×3 │ ← Bộ lọc (kernel) nhỏ trượt qua ảnh
  │  └─────┘
  │
  ▼
Lớp 1: Phát hiện cạnh, vân (32 bộ lọc)
  │
  ▼
Lớp 2: Phát hiện hình dạng đơn giản (64 bộ lọc)
  │
  ▼
Lớp 3: Phát hiện pattern phức tạp (128 bộ lọc)
  │
  ▼
Lớp 4: Phát hiện đặc trưng cấp cao (256 bộ lọc)
  │
  ▼
Fully Connected: Phân loại thành 10 lớp
```

**Ví dụ**:
- Lớp 1 có thể học nhận biết "vạch ngang" (harmonic) trong spectrogram
- Lớp 2 kết hợp thành "nhóm vạch" (formant)
- Lớp 3 nhận biết "pattern tiếng sủa" (burst + silence + burst)
- Lớp 4 phân biệt "chó sủa" vs "còi xe"

### 4.3 Các thành phần trong CNN

**Conv2d (Tích chập 2D)**:
- Kernel 3×3 trượt qua ảnh, tính tích chập
- Output = "feature map" — ảnh mới highlight các pattern

```
Input:                Kernel 3×3:         Output:
[1 2 3 4 5]          [1 0 -1]           Highlight
[2 3 4 5 6]    *     [1 0 -1]     =     các cạnh
[3 4 5 6 7]          [1 0 -1]           dọc
```

**BatchNorm (Chuẩn hóa theo batch)**:
- Chuẩn hóa output của mỗi lớp → training ổn định hơn, nhanh hội tụ hơn

**ReLU (Hàm kích hoạt)**:
- f(x) = max(0, x)
- Giữ giá trị dương, bỏ giá trị âm → thêm tính phi tuyến

**MaxPool2d (Gộp tối đa)**:
- Giảm kích thước ảnh xuống 1/2 → giảm tính toán, giữ feature quan trọng nhất
- Lấy giá trị lớn nhất trong mỗi vùng 2×2

```
[1 3 | 2 4]        MaxPool 2×2       [3 | 4]
[5 2 | 6 1]        ──────────→       [5 | 6]
[─────────]
[3 5 | 1 2]                          [5 | 2]
[4 1 | 2 0]
```

**AdaptiveAvgPool2d(1)**:
- Gộp toàn bộ feature map thành 1 số (trung bình toàn bộ)
- Cho phép CNN nhận input kích thước bất kỳ

**Dropout**:
- Ngẫu nhiên "tắt" một số neuron trong lúc training (50% hoặc 30%)
- Ngăn overfit — mạng không thể phụ thuộc vào bất kỳ neuron đơn lẻ nào

### 4.4 Kiến trúc CNN trong dự án

File: `src/models/deep_learning.py`

```
Input: (1, 128, 173) — 1 kênh × 128 mel bands × 173 time frames

BLOCK 1: Conv2d(1→32, 3×3)   → BatchNorm(32) → ReLU → MaxPool(2)
          Output: (32, 64, 86)

BLOCK 2: Conv2d(32→64, 3×3)  → BatchNorm(64) → ReLU → MaxPool(2)
          Output: (64, 32, 43)

BLOCK 3: Conv2d(64→128, 3×3) → BatchNorm(128) → ReLU → MaxPool(2)
          Output: (128, 16, 21)

BLOCK 4: Conv2d(128→256, 3×3) → BatchNorm(256) → ReLU → AdaptiveAvgPool(1)
          Output: (256, 1, 1) → flatten → (256,)

CLASSIFIER:
  Dropout(0.5) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→10)
  Output: (10,) — xác suất cho 10 lớp
```

### 4.5 Quá trình training

```python
import torch
import torch.nn as nn

model = CNN2D()                                    # Tạo model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
criterion = nn.CrossEntropyLoss()                  # Hàm loss cho phân loại

for epoch in range(30):
    for X_batch, y_batch in train_loader:
        # Forward: đưa data qua model → dự đoán
        predictions = model(X_batch)

        # Tính loss (sai số)
        loss = criterion(predictions, y_batch)

        # Backward: tính gradient (đạo hàm loss theo trọng số)
        loss.backward()

        # Update: cập nhật trọng số để giảm loss
        optimizer.step()
        optimizer.zero_grad()
```

**Early stopping**: Nếu val_loss không giảm sau 5 epoch → dừng training để tránh overfit.

**ReduceLROnPlateau**: Nếu val_loss không giảm sau 3 epoch → giảm learning rate xuống 1/2 → "bước nhỏ hơn" để tìm minimum tốt hơn.

### 4.6 Tại sao CNN kém hơn trong dự án?

CNN đạt 66.7% — thấp hơn RF (71.5%). Lý do:

1. **Dataset quá nhỏ**: ~7,800 mẫu train/fold. CNN cần hàng chục ngàn mẫu để học tốt
2. **Không data augmentation**: Không xoay/dịch/thêm nhiễu vào spectrogram → CNN thấy ít mẫu
3. **Early stopping (patience=5)**: Nhiều fold dừng ở epoch 9-16, chưa hội tụ đủ
4. **Mel spectrogram mất thông tin**: Chuyển từ waveform → mel spectrogram mất thông tin phase

---

## 5. Scikit-Learn trong dự án

### 5.1 Scikit-Learn là gì?

**scikit-learn** (sklearn) là thư viện Python phổ biến nhất cho Machine Learning truyền thống. Nó cung cấp:
- Các thuật toán ML (SVM, RF, KNN, Logistic Regression...)
- Tiền xử lý dữ liệu (scaling, PCA, encoding...)
- Đánh giá model (accuracy, F1, confusion matrix, cross-validation...)
- Pipeline (nối các bước lại thành 1 luồng)

### 5.2 Các module sklearn dùng trong dự án

#### `sklearn.svm.SVC` — Support Vector Classifier

```python
from sklearn.svm import SVC

# Tạo SVM classifier
svm = SVC(
    kernel='rbf',       # Kernel RBF (phi tuyến)
    C=10,               # Regularization: cân bằng giữa margin lớn và phân loại đúng
    gamma='scale',      # gamma = 1 / (n_features * X.var())
    random_state=42     # Seed để tái tạo kết quả
)
svm.fit(X_train, y_train)      # Train
preds = svm.predict(X_test)    # Dự đoán
```

#### `sklearn.ensemble.RandomForestClassifier` — Rừng Ngẫu Nhiên

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,   # 500 cây quyết định
    max_depth=None,     # Cây phát triển tối đa
    random_state=42,    # Seed
    n_jobs=-1           # Dùng tất cả CPU cores
)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

# Xem feature nào quan trọng
rf.feature_importances_  # array shape (931,)
```

#### `sklearn.preprocessing.StandardScaler` — Chuẩn hóa dữ liệu

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# fit_transform = tính mean, std từ X_train rồi biến đổi
# Mỗi feature: (x - mean) / std → mean=0, std=1

X_test_scaled = scaler.transform(X_test)
# transform = dùng mean, std đã tính từ train (KHÔNG fit lại trên test!)
```

**Tại sao cần chuẩn hóa?**

Nếu feature A có range [0, 10000] và feature B có range [0, 1]:
- SVM sẽ bị "chi phối" bởi feature A (vì nó có giá trị lớn hơn nhiều)
- Sau chuẩn hóa: cả 2 đều có range tương đương → công bằng

#### `sklearn.decomposition.PCA` — Giảm chiều

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=200)
X_reduced = pca.fit_transform(X_scaled)  # (7800, 931) → (7800, 200)

# PCA tìm 200 "hướng" chứa nhiều variance nhất
# Giữ ~95% thông tin, bỏ 731 chiều dư thừa/nhiễu
```

**Hình dung PCA**: Nếu dữ liệu 3D nằm trên 1 mặt phẳng, PCA "chiếu" xuống 2D mà không mất thông tin quan trọng.

#### `sklearn.model_selection.GridSearchCV` — Tìm tham số tối ưu

```python
from sklearn.model_selection import GridSearchCV

# Thử tất cả tổ hợp tham số
param_grid = {
    'C': [1, 10],
    'gamma': ['scale']
}
# Tổ hợp: C=1+gamma=scale, C=10+gamma=scale → 2 tổ hợp

grid = GridSearchCV(
    SVC(kernel='rbf'),
    param_grid,
    cv=3,              # 3-fold cross-validation bên trong
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)

grid.best_params_      # {'C': 10, 'gamma': 'scale'}
grid.best_estimator_   # Model đã train với tham số tốt nhất
```

**Lưu ý**: Trong dự án cuối, mình bỏ GridSearchCV cho SVM vì quá chậm (931 features × 7800 samples), dùng tham số cố định C=10 thay thế.

#### `sklearn.metrics` — Đánh giá model

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Accuracy: % dự đoán đúng
accuracy = accuracy_score(y_true, y_pred)
# Ví dụ: 700 đúng / 1000 tổng = 70%

# F1 Score (macro): trung bình F1 của tất cả lớp
f1 = f1_score(y_true, y_pred, average='macro')
# F1 = 2 * (precision * recall) / (precision + recall)
# 'macro' = tính F1 cho mỗi lớp rồi lấy trung bình (công bằng cho lớp nhỏ)

# Confusion Matrix: ma trận nhầm lẫn
cm = confusion_matrix(y_true, y_pred)
# cm[i][j] = số mẫu lớp i bị dự đoán thành lớp j
```

**Giải thích Precision, Recall, F1**:

```
Ví dụ cho lớp "dog_bark":

                    Dự đoán: dog_bark    Dự đoán: khác
Thật: dog_bark         80 (TP)              20 (FN)
Thật: khác             10 (FP)             890 (TN)

Precision = TP / (TP + FP) = 80/90 = 88.9%
→ "Trong những cái model nói là dog_bark, bao nhiêu % đúng?"

Recall = TP / (TP + FN) = 80/100 = 80.0%
→ "Trong tất cả dog_bark thật, model tìm ra bao nhiêu %?"

F1 = 2 × (88.9% × 80.0%) / (88.9% + 80.0%) = 84.2%
→ Trung bình hài hòa của Precision và Recall
```

#### `sklearn.pipeline.Pipeline` — Nối các bước

```python
from sklearn.pipeline import Pipeline

# Tạo pipeline: Scale → SVM (tự động chạy tuần tự)
pipe = Pipeline([
    ('scaler', StandardScaler()),   # Bước 1: chuẩn hóa
    ('svm', SVC(kernel='rbf'))      # Bước 2: SVM
])

pipe.fit(X_train, y_train)    # Tự động: scale → train SVM
pipe.predict(X_test)          # Tự động: scale → predict
```

### 5.3 Tổng hợp sklearn modules

| Module | Chức năng | Dùng ở đâu |
|--------|----------|-------------|
| `sklearn.svm.SVC` | Phân loại SVM | Train SVM classifier |
| `sklearn.ensemble.RandomForestClassifier` | Phân loại Random Forest | Train RF classifier |
| `sklearn.preprocessing.StandardScaler` | Chuẩn hóa mean=0, std=1 | Trước khi đưa vào SVM/RF |
| `sklearn.decomposition.PCA` | Giảm chiều | 931 → 200 cho SVM |
| `sklearn.model_selection.GridSearchCV` | Tìm hyperparameters | Tối ưu C, gamma cho SVM |
| `sklearn.metrics.accuracy_score` | Tính accuracy | Đánh giá mỗi fold |
| `sklearn.metrics.f1_score` | Tính F1 score | Đánh giá mỗi fold |
| `sklearn.metrics.confusion_matrix` | Ma trận nhầm lẫn | Phân tích lỗi |
| `sklearn.metrics.classification_report` | Báo cáo per-class | Phân tích từng lớp |
| `sklearn.pipeline.Pipeline` | Nối bước xử lý | Scale + SVM thành 1 luồng |

---

## 6. So sánh 3 model

### 6.1 Bảng so sánh chi tiết

| Tiêu chí | SVM | Random Forest | CNN-2D |
|----------|-----|---------------|--------|
| **Loại** | Classical ML | Classical ML | Deep Learning |
| **Thư viện** | scikit-learn | scikit-learn | PyTorch |
| **Input** | Vector 200 số (sau PCA) | Vector 931 số | Ảnh 128×173 |
| **Cách hoạt động** | Tìm đường ranh giới tối ưu | 500 cây vote | Học pattern từ ảnh |
| **Training time/fold** | ~45 giây | ~5 giây | ~10-25 phút |
| **Accuracy (Pipeline A)** | 70.1% | **71.5%** | 66.7% |
| **Accuracy (Pipeline B)** | 70.0% | 71.2% | 67.6% |
| **Variance (CI)** | ±3.2% | **±2.6%** (thấp nhất) | ±5.2% (cao nhất) |
| **Cần GPU** | Không | Không | Có (MPS/CUDA) |
| **Cần chuẩn hóa** | Bắt buộc | Không bắt buộc | Không |
| **Cần PCA** | Có (vì chậm) | Không | Không |
| **Feature importance** | Không | **Có** | Không trực tiếp |
| **Phù hợp dataset nhỏ** | Tốt | **Rất tốt** | Kém |

### 6.2 Khi nào dùng model nào?

- **Random Forest**: Dataset nhỏ-vừa, cần nhanh, cần biết feature importance → **chọn RF**
- **SVM**: Dataset vừa, dữ liệu đã chuẩn hóa tốt, muốn model đơn giản → chọn SVM
- **CNN**: Dataset **lớn** (>50,000 mẫu), có GPU, có data augmentation → chọn CNN

### 6.3 Kết luận về model

Trong dự án DSP501 với ~8,700 mẫu:

> **Random Forest thắng** vì dataset nhỏ, features tốt (931 chiều MFCC+spectral), và RF xử lý tốt dữ liệu cao chiều mà không cần PCA. CNN thua vì thiếu dữ liệu và không có data augmentation.
