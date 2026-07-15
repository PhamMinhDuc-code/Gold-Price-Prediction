# Gold Price Prediction System

Hệ thống dự báo giá vàng (XAU) dựa trên dữ liệu OHLCV lịch sử (2004–2026) kết hợp các
chỉ số kinh tế (USD Index - DXY, giá Dầu, lợi suất Trái phiếu Kho bạc Mỹ kỳ hạn 10 năm).
Hỗ trợ 4 kiến trúc mô hình: LSTM, GRU, XGBoost, Random Forest.

> ⚠️ Dự án phục vụ mục đích học tập/nghiên cứu. Các dự báo và số liệu trong báo cáo
> **không phải là khuyến nghị đầu tư/giao dịch tài chính**.

---

## 1. Yêu cầu hệ thống

- Python **3.11** (khuyến nghị — tránh 3.13+ vì TensorFlow chưa hỗ trợ chính thức)
- Conda (Anaconda/Miniconda) hoặc `venv` chuẩn của Python
- Git

---

## 2. Cài đặt môi trường ảo

Chọn **1 trong 2 cách** dưới đây (khuyên dùng Conda nếu máy đã cài sẵn).

### Cách A — Dùng Conda (khuyên dùng)

```bash
# Tạo môi trường ảo riêng cho project, Python 3.11
conda create -n gold-pred python=3.11 -y

# Kích hoạt môi trường
conda activate gold-pred
```

### Cách B — Dùng venv chuẩn của Python

```bash
# Tại thư mục gốc project
python -m venv .venv

# Kích hoạt môi trường
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate
```

Sau khi kích hoạt, dấu nhắc lệnh (prompt) sẽ hiện tên môi trường ở đầu dòng, ví dụ
`(gold-pred) PS D:\...>` hoặc `(.venv) $`.

### Cài dependencies

```bash
cd gold-price-prediction     # vào đúng thư mục project vừa clone
pip install -r requirements.txt
```

### Đăng ký kernel Jupyter cho VS Code

```bash
python -m ipykernel install --user --name gold-pred --display-name "Python (gold-pred)"
```

Trong VS Code, mở bất kỳ notebook nào trong `notebooks/` → góc trên phải chọn kernel
**"Python (gold-pred)"** → **Restart Kernel** trước khi chạy.

---

## 3. Cấu trúc thư mục

```
gold-price-prediction/
├── .kiro/specs/            # Tài liệu spec (requirements, design, tasks)
├── config.py                # Cấu hình toàn hệ thống (đường dẫn, hyperparameters...)
├── requirements.txt          # Danh sách thư viện cần cài
├── data/                     # Dữ liệu gốc (CSV giá vàng) — tự thêm, xem mục 4
├── src/                      # Toàn bộ mã nguồn (ingestion, preprocessing, training...)
├── notebooks/                # 3 notebook chính (xem mục 5)
├── models/                   # Model đã train (tự sinh, không commit lên Git)
├── reports/                  # Báo cáo/kết quả (tự sinh, xem mục 6)
└── logs/                     # File log hệ thống (tự sinh)
```

---

## 4. Chuẩn bị dữ liệu

Đặt file dữ liệu giá vàng dạng CSV (cột `Date;Open;High;Low;Close;Volume`) vào:
```
data/XAU_1d_data.csv
```
(Nếu bạn nhận project từ người khác, có thể họ đã gửi kèm file này riêng do `.gitignore`
loại trừ CSV ra khỏi repo.)

---

## 5. Chạy notebook — thứ tự thực hiện

Mở lần lượt 3 notebook trong thư mục `notebooks/`, chọn kernel `gold-pred`,
**Restart Kernel** rồi **Run All**:

| Thứ tự | Notebook | Mục đích |
|---|---|---|
| 1 | `01_data_exploration.ipynb` | Khám phá, kiểm tra chất lượng dữ liệu |
| 2 | `02_model_training.ipynb` | Tiền xử lý, feature engineering, train & đánh giá model, lưu model |
| 3 | `03_prediction_and_forecasting.ipynb` | Tải model đã train, dự báo giá vàng 7/14/30 ngày, tạo báo cáo |

> Notebook 2 phải chạy **trước** notebook 3 (vì notebook 3 cần model đã được lưu vào
> `models/` từ bước train).

---

## 6. Xem kết quả — báo cáo, biểu đồ, HTML

Sau khi chạy xong, kết quả nằm trong thư mục **`reports/`**. Có 3 dạng file, cách xem
khác nhau tùy loại:

### 6.1. Báo cáo HTML (dễ xem nhất — có giao diện, biểu đồ nhúng sẵn)

File dạng `*_report.html` (ví dụ `notebook_prediction_report.html`,
`demo_viz_comprehensive_report.html`). Cách mở:

- **Trong VS Code**: click chuột phải vào file `.html` trong khung Explorer bên trái →
  chọn **"Open with Live Server"** (cần cài extension *Live Server* nếu chưa có), hoặc
  đơn giản nhất: click phải → **"Reveal in File Explorer"** rồi double-click file đó,
  Windows sẽ tự mở bằng trình duyệt mặc định (Chrome/Edge...).
- **Ngoài VS Code**: vào thư mục `reports/` bằng File Explorer, double-click file `.html`
  bất kỳ → mở bằng trình duyệt, xem đầy đủ metrics + biểu đồ + link CSV chi tiết.

### 6.2. Ảnh biểu đồ (`.png`)

Ví dụ `notebook_prediction_report_plot_1.png`, `demo_viz_feature_importance.png`.
Trong VS Code, click trực tiếp vào file `.png` trong Explorer để xem preview ngay
trong tab (không cần mở app ngoài).

### 6.3. Dữ liệu chi tiết (`.csv`, `.json`)

- `*_predictions.csv`: bảng dự đoán chi tiết theo ngày — mở bằng Excel, hoặc trong
  VS Code cài extension **Excel Viewer** / **Edit csv** để xem dạng bảng ngay trong IDE.
- `*.json`: chứa toàn bộ metrics dạng máy đọc được (dùng nếu muốn tích hợp/tự động hóa
  tiếp) — mở bằng VS Code là đủ, có format sẵn dễ đọc.

### 6.4. Cách nhanh nhất nếu muốn gửi cho người khác xem (không cần code)

Chỉ cần gửi **2 file**: file `.html` (báo cáo) + file `.png` cùng tên đứng cạnh nó
(ảnh biểu đồ mà HTML nhúng vào) — giữ **chung 1 thư mục** khi gửi, vì file HTML tham
chiếu ảnh bằng đường dẫn tương đối, tách rời ra ảnh sẽ không hiện được trong HTML.

---

## 7. Các lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Sai kernel/môi trường | Kiểm tra `conda activate gold-pred` đã chạy, chọn đúng kernel trong VS Code |
| `ModuleNotFoundError: No module named 'tensorflow'` | Thiếu set biến môi trường JAX backend | Đảm bảo dòng `os.environ['KERAS_BACKEND'] = 'jax'` chạy **trước** mọi import keras |
| `KeyError: 'Date'` | Nhầm giữa Date là cột và Date là index | Đã sửa trong notebook — dùng `df.index` thay vì `df['Date']` |

---

## 8. Công nghệ sử dụng

- **Xử lý dữ liệu**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: Keras 3 (JAX backend)
- **Trực quan hóa**: matplotlib, seaborn
- **Dữ liệu kinh tế**: yfinance

---

## 9. Giấy phép & Miễn trừ trách nhiệm

Dự án phục vụ mục đích học tập. Mọi dự báo giá vàng chỉ mang tính minh họa cho pipeline
machine learning, **không cấu thành lời khuyên tài chính**. Người dùng tự chịu trách
nhiệm với mọi quyết định đầu tư dựa trên thông tin từ hệ thống này.
