# Q2: Huấn luyện mô hình dự đoán khoảng giá

## Mô tả
Phần này xây dựng và huấn luyện các mô hình dự đoán khoảng giá nhà (prediction intervals) với độ tin cậy 90%.

## Cấu trúc thư mục
Q2/
├── README.md
├── model_training.ipynb
├── submission.csv
└── requirements.txt

## Cách chạy trên Google Colab

1. Upload file `model_training.ipynb` lên Google Colab
2. Đảm bảo đã chạy xong phần Q1 hoặc có dữ liệu đã được xử lý
3. Chạy từng cell theo thứ tự

## Các mô hình được sử dụng

### 1. Baseline Model
- Linear Quantile Regression
- Đơn giản, dễ hiểu
- Làm baseline để so sánh

### 2. LightGBM
- Gradient Boosting với quantile objective
- Xử lý tốt categorical features
- Hiệu suất cao, training nhanh

### 3. XGBoost
- Extreme Gradient Boosting
- Quantile regression capability
- Robust và ổn định

### 4. Ensemble Model
- Kết hợp LightGBM và XGBoost
- Cải thiện độ ổn định
- Giảm overfitting

## Phương pháp đánh giá

### Winkler Score
- Metric chính của competition
- Cân bằng giữa độ rộng interval và coverage
- Công thức: `mean(width) + (2/α) * mean(penalties)`

### Coverage Score
- Tỷ lệ actual values nằm trong prediction intervals
- Target: 90% coverage

## Kết quả

| Model | Winkler Score | Coverage |
|-------|---------------|----------|
| Linear Regression | XXX.XX | 0.XXX |
| LightGBM | XXX.XX | 0.XXX |
| XGBoost | XXX.XX | 0.XXX |
| Ensemble | XXX.XX | 0.XXX |

## Files đầu ra
- `submission.csv`: File submit lên Kaggle
- Model performance metrics
- Feature importance analysis

## Requirements
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.40.0 (optional)

## Lưu ý
- Đảm bảo có đủ RAM khi training (recommend >= 12GB)
- Có thể cần GPU để tăng tốc training
- Kiểm tra format của submission file trước khi submit
