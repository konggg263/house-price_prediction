# Q1: EDA và Xử lý dữ liệu (Exploratory Data Analysis & Data Preprocessing)

## Mô tả
Phần này thực hiện phân tích khám phá dữ liệu và xử lý dữ liệu ban đầu cho bài toán dự đoán giá nhà.

## Cấu trúc thư mục

```
┌─ 📊 **Q1** - Exploratory Data Analysis & Preprocessing
├── 📁 data/
│   ├── 💾 dataset.csv
│   ├── 📄 sample_submission.csv
│   └── 🔬 test.csv
├── 📁 processed_data/                  # Sinh ra sau khi chạy xong eda_preprocessing.ipynb
├── 📚 README.md
├── 🔍 eda_preprocessing.ipynb
└── ⚙️ requirements.txt
```

## Cách chạy trên Google Colab

1. Upload file `eda_preprocessing.ipynb` lên Google Colab
2. Thêm cell này vào đầu:
```
from google.colab import drive
drive.mount('/content/drive')
```

3. Tải dataset từ Kaggle competition: "Prediction Interval Competition II - House Price" hoặc upload dataset từ Q1 lên folder Colab Notebooks trong Google Drive
4. Tại cell load dữ liệu, thay thế các đường dẫn đến các file .csv phù hợp

```
# Load dữ liệu
train = pd.read_csv("/content/drive/My Drive/Colab Notebooks/data/dataset.csv")             # Thay đổi đường dẫn nếu cần
test = pd.read_csv("/content/drive/My Drive/Colab Notebooks/data/test.csv")                 # Thay đổi đường dẫn nếu cần
sample = pd.read_csv("/content/drive/My Drive/Colab Notebooks/data/sample_submission.csv")  # Thay đổi đường dẫn nếu cần
```

Tại cell LƯU DỮ LIỆU ĐÃ XỬ LÝ:

```
# Tạo thư mục processed_data nếu chưa có
import os
os.makedirs('/content/drive/My Drive/Colab Notebooks/processed_data', exist_ok=True)

# Lưu dữ liệu chính
X_train.to_csv('/content/drive/My Drive/Colab Notebooks/processed_data/X_train.csv', index=False)
X_test.to_csv('/content/drive/My Drive/Colab Notebooks/processed_data/X_test.csv', index=False)

if y_train is not None:
    y_train.to_csv('/content/drive/My Drive/Colab Notebooks/processed_data/y_train.csv', index=False)

# Lưu test ids để tạo submission
test_ids = test['id']
test_ids.to_csv('/content/drive/My Drive/Colab Notebooks/processed_data/test_ids.csv', index=False)

# Lưu metadata
metadata = {
    'train_shape': X_train.shape,
    'test_shape': X_test.shape,
    'features': list(X_train.columns),
    'categorical_features': list(label_encoders.keys()),
    'target_stats': {
        'mean': float(y_train.mean()) if y_train is not None else None,
        'std': float(y_train.std()) if y_train is not None else None,
        'min': float(y_train.min()) if y_train is not None else None,
        'max': float(y_train.max()) if y_train is not None else None
    }
}

import json
with open('/content/drive/My Drive/Colab Notebooks/processed_data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Lưu encoders và scaler
with open('/content/drive/My Drive/Colab Notebooks/processed_data/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('/content/drive/My Drive/Colab Notebooks/processed_data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Lưu dữ liệu gốc đã feature engineering (trước khi encode và scale)
train_fe.to_csv('/content/drive/My Drive/Colab Notebooks/processed_data/train_with_features.csv', index=False)
test_fe.to_csv('/content/drive/My Drive/Colab Notebooks/processed_data/test_with_features.csv', index=False)
```

5. Chạy từng cell theo thứ tự

## Các bước thực hiện

### 1. Phân tích mô tả cơ bản
- Thống kê mô tả các thuộc tính
- Phân tích phân bố giá nhà
- Phát hiện missing values

### 2. Phân tích trực quan
- Histogram và box plot của giá nhà
- Correlation matrix
- Phân tích theo thành phố, năm xây dựng
- Scatter plots của các biến quan trọng

### 3. Xử lý dữ liệu
- Phát hiện và xử lý outliers
- Feature engineering
- Chuẩn hóa dữ liệu
- Encoding categorical variables

## Kết quả
- Dataset được làm sạch và chuẩn bị cho modeling
- Các insights quan trọng về dữ liệu
- Features mới được tạo để cải thiện model performance