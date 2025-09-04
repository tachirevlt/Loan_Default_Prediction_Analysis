
# Dự án Phân tích và Dự đoán Rủi ro Tín dụng với XGBoost

## Mô tả Dự án
Dự án này tập trung vào việc phân tích dữ liệu tín dụng và xây dựng một mô hình dự đoán rủi ro (Default) sử dụng thuật toán XGBoost. Dữ liệu được xử lý, trực quan hóa, và tối ưu hóa để đạt hiệu suất tốt nhất với các kỹ thuật hiện đại.

## Nội dung Chính

### 1. Cài đặt và Chuẩn bị
- **Thư viện cần thiết:** Dự án sử dụng các thư viện Python như `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, và `xgboost` để phân tích dữ liệu, trực quan hóa, và huấn luyện mô hình.
- **Cách cài đặt:**
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
  ```

### 2. Tiền xử lý Dữ liệu
- **Đọc dữ liệu:** Tải dữ liệu từ file CSV và kiểm tra các giá trị thiếu (`missing values`) hoặc trùng lặp (`duplicates`).
- **Xử lý cột:** Loại bỏ các cột không cần thiết để tối ưu hóa dữ liệu.
- **Kiểm tra ban đầu:** Đảm bảo dữ liệu sạch trước khi phân tích.

### 3. Phân tích Khám phá (EDA)
- **Phân loại biến:** Chia dữ liệu thành biến số (`numeric`) và biến hạng mục (`categorical`).
- **Trực quan hóa:**
  - **Histogram:** Hiển thị phân phối của các biến số.
  - **Pie Chart:** Thể hiện tỷ lệ của các biến hạng mục.
- **Mục đích:** Hiểu rõ đặc điểm dữ liệu và chuẩn bị cho phân tích sâu hơn.

### 4. Phân tích Tương quan
- **Tương quan với Target (Default):** Đánh giá mối quan hệ giữa các biến và biến mục tiêu, bao gồm ý nghĩa thống kê (p-value) và ý nghĩa thực tế (correlation strength).
- **Tương quan giữa các biến:** Kiểm tra xem có đa cộng tuyến (`multicollinearity`) không.
- **Nhận xét:** Không có tương quan mạnh hoặc biến không ý nghĩa để loại bỏ.

### 5. Xử lý Dữ liệu Nâng cao
- **Loại bỏ Outlier:** Sử dụng phương pháp IQR (Interquartile Range) để loại bỏ các giá trị ngoại lai.
- **Mã hóa biến:** Áp dụng one-hot encoding cho các biến categorical để phù hợp với mô hình.

### 6. Chuẩn bị Dữ liệu cho Mô hình
- **Chia tập dữ liệu:** Phân chia thành tập huấn luyện (`train`) và tập kiểm tra (`test`).
- **Cân bằng dữ liệu:** Sử dụng SMOTE (Synthetic Minority Over-sampling Technique) để xử lý dữ liệu không cân bằng.

### 7. Xây dựng và Tối ưu Mô hình
- **Mô hình:** Sử dụng XGBoost để dự đoán rủi ro tín dụng.
- **Tối ưu hóa:** Áp dụng GridSearchCV để tìm ra tập hyperparameter tốt nhất.(bị lỗi xung đột thư viện khi sử dụng visual code studio)
- **Đánh giá:**
  - Sử dụng Classification Report để xem độ chính xác, precision, recall, và F1-score.
  - Tính AUC (Area Under the Curve) để đánh giá khả năng phân loại.
- **Lưu ý:** Không loại bỏ feature nào vì không phát hiện đa cộng tuyến mạnh hoặc feature không ý nghĩa.

### 8. Kết quả và Tương lai
- Dự án cung cấp một mô hình cơ bản để dự đoán rủi ro tín dụng với hiệu suất tốt nhờ tối ưu hóa hyperparameter.
- **Tương lai:** Có thể mở rộng bằng cách thêm feature engineering, thử nghiệm các mô hình khác (như LightGBM), hoặc cải thiện xử lý dữ liệu không cân bằng.

## Cách Sử dụng
1. Clone repository:
   ```bash
   git clone <repository-url>
   ```
2. Cài đặt các thư viện cần thiết (xem phần Cài đặt).
3. Chạy file chính (ví dụ: `main.py`) để thực hiện toàn bộ quy trình.
4. Kiểm tra kết quả trong output hoặc file được lưu (nếu có).

## Yêu cầu Hệ thống
- Python 3.8+
- Các thư viện đã liệt kê ở phần Cài đặt.

## Tác giả
[![GitHub](https://img.shields.io/badge/GitHub-tachirevlt-blue?logo=github)](https://github.com/tachirevlt)

## Cảm ơn
Cảm ơn đã sử dụng dự án này! Nếu có câu hỏi hoặc đóng góp, vui lòng mở issue trên GitHub hoặc liên hệ trực tiếp.
```

---
