# Q1: EDA và Xử lý Dữ liệu

## Mô tả
Phần này thực hiện phân tích khám phá dữ liệu (EDA) và xử lý dữ liệu cho bài toán dự đoán khoảng giá nhà.

## Cấu trúc Files
```
Q1/
├── data_analysis.ipynb      # Notebook chính
├── eda_preprocessing.py     # Class và functions
├── README.md               # File này
└── processed_data/         # Dữ liệu đã xử lý (output)
    ├── X_train_processed.csv
    ├── X_test_processed.csv
    ├── y_train.csv
    └── scaler.pkl
```

## Yêu cầu Hệ thống
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy
- jupyter notebook

## Cách chạy trên Google Colab

### Bước 1: Upload dữ liệu
```python
from google.colab import files

# Upload train.csv và test.csv từ Kaggle
uploaded = files.upload()
```

### Bước 2: Cài đặt thư viện
```python
!pip install lightgbm xgboost seaborn
```

### Bước 3: Chạy notebook
```python
# Copy notebook code vào Colab và chạy từng cell
# Hoặc upload file .ipynb
```

## Các bước thực hiện

### 1. Tải và khám phá dữ liệu
- Load train.csv và test.csv
- Thống kê mô tả cơ bản
- Phân tích missing values
- Kiểm tra kiểu dữ liệu

### 2. Phân tích phân phối SalePrice
- Histogram và density plot
- Log transformation
- Q-Q plot để kiểm tra normality
- Phát hiện outliers

### 3. Phân tích tương quan
- Correlation heatmap
- Top features có correlation cao với SalePrice
- Scatter plots cho các features quan trọng

### 4. Xử lý dữ liệu thiếu
- Numeric columns: Fill bằng median
- Categorical columns: Fill bằng mode
- Kiểm tra sau khi xử lý

### 5. Feature Engineering
- Tạo TotalSF, TotalArea
- Tính HouseAge, RemodAge
- Tổng số phòng tắm
- Interaction features

### 6. Xử lý Outliers
- Phát hiện bằng IQR method
- Visualize bằng box plots
- Quyết định có loại bỏ hay không

### 7. Encoding Categorical Variables
- One-hot encoding cho ít categories
- Label encoding cho nhiều categories
- Đảm bảo consistency giữa train/test

### 8. Feature Scaling
- RobustScaler cho outliers
- Fit trên train, transform cả train/test
- Lưu scaler để dùng sau

## Output
- `X_train_processed.csv`: Features đã xử lý cho training
- `X_test_processed.csv`: Features đã xử lý cho testing  
- `y_train.csv`: Target variable
- `scaler.pkl`: Fitted scaler object

## Kết quả chính
- Dataset shape: [Sẽ được cập nhật khi chạy]
- Top 5 features quan trọng: [Sẽ được cập nhật]
- Missing values: Đã xử lý hoàn toàn
- Outliers: Đã phân tích và xử lý

## Troubleshooting
- **Lỗi import**: Cài đặt thư viện bị thiếu
- **Memory error**: Giảm kích thước dataset hoặc dùng chunking
- **File not found**: Kiểm tra đường dẫn file CSV