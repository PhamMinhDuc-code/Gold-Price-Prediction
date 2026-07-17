# Phân tích & Dự đoán Giá Vàng (Thế giới + Việt Nam)

> **Câu hỏi nghiên cứu:** Có nên mua vàng ở thời điểm hiện tại hay không?

Cập nhật: 17/07/2026 — dữ liệu phủ 2004 → 17/07/2026 (thế giới), 2010 → 17/07/2026 (Việt Nam).

---

## 1. Tổng quan

Dự án dự báo giá vàng theo kiến trúc **2 tầng**:

1. **Tầng 1 — Giá vàng thế giới** (log-return XAU/USD, chuỗi chính LBMA AM/PM fix), dùng LightGBM/XGBoost làm model chính, SARIMAX làm model thống kê đối chiếu, Naive + Drift làm baseline bắt buộc, LSTM/GRU là lựa chọn so sánh thêm nếu còn thời gian.
2. **Tầng 2 — Premium vàng SJC Việt Nam** so với giá quy đổi từ thế giới (`premium = SJC_bán − giá_quy_đổi`), là chuỗi mean-reverting có structural break, mô hình hóa riêng bằng OLS/GLS hoặc LightGBM nhỏ với các biến dummy chính sách.

Kết quả cuối không phải một con số dự báo đơn lẻ, mà là khung khuyến nghị 3 trụ: tín hiệu định lượng, định giá tương đối (premium + real yield), và kịch bản rủi ro theo lãi suất Fed.

## 2. Cấu trúc dữ liệu (`data/`)

Dữ liệu chia theo nguồn gốc: `vietnam/` (nguồn trong nước) và `world/` (nguồn quốc tế).

### 2.1 `world/` — Dữ liệu thế giới

| Nhóm | File chính | Nội dung | Phủ thời gian |
|---|---|---|---|
| Giá vàng & thị trường | `gold_market/lbma_gold_am_pm_history.csv` | Giá XAU/USD chốt London AM/PM — **chuỗi giá chính** | 1968 → 16/07/2026 |
| | `gold_market/market_daily_2004_present.csv` | GC=F, SI=F, CL=F, ^GSPC, ^VIX, DX-Y.NYB, GLD, USDVND=X (yfinance) — **file thị trường chính** | 2004 → 17/07/2026 |
| | `gold_market/gld_shares_outstanding.csv` | Chứng chỉ quỹ GLD lưu hành (SEC EDGAR) — đo dòng tiền tổ chức | 2010 → 07/2026 |
| Vĩ mô Mỹ (FRED) | `macro_fred/fred_series_full_history.csv` | 14 series hợp nhất: DGS10, DFII10 (real yield — biến quan trọng nhất với vàng), DFF, CPIAUCSL, DTWEXBGS, VIXCLS, DCOILWTICO, T10YIE, NFCI, STLFSI2, AAA10Y/BAA10Y, M2SL | 1947 → 14/07/2026 |
| | `macro_fred/t10yie_computed_2003_present.csv` | Breakeven inflation tự tính (DGS10 − DFII10), kéo dài về 2003 | 2003 → 14/07/2026 |
| Rủi ro địa chính trị | `geopolitical_risk/gpr_daily_geopolitical_risk.csv` | Chỉ số GPR (Caldara & Iacoviello) — vàng thường tăng khi GPR đột biến | 1985 → 13/07/2026 |

⚠️ `STLFSI2` ngừng cập nhật từ 01/2022 (thay bằng STLFSI4) — loại khỏi model hoặc tải bổ sung thủ công.

### 2.2 `vietnam/` — Dữ liệu Việt Nam

| Nhóm | File chính | Nội dung | Phủ thời gian |
|---|---|---|---|
| Giá vàng trong nước | `gold_prices/gold_raw_history_all_sources_2010_2026.csv` | Giá mua/bán SJC (và PNJ), 2 archive chéo, mọi lần đổi giá trong ngày — **file chính** | 02/01/2010 → 07/07/2026 |
| | `gold_prices/gold_recent_update_pnj_sjc.csv` | Cập nhật mới nhất từ API chính thức PNJ (nguồn đối chiếu thứ 3) | 01/07 → 17/07/2026 |
| | `gold_prices/canonical_gold_products_daily.csv` | Giá daily chuẩn hóa theo sản phẩm (miếng SJC 1L, nhẫn 9999) | 2010 → 11/07/2026 |
| | `gold_prices/data_quality_report.md`, `quality_flags.csv` | Báo cáo kiểm định chất lượng: 0 vi phạm sell>buy, lệch chéo nguồn TB 0,023% | — |
| Tỷ giá | `fx_rates/vcb_usd_daily.csv` | Tỷ giá bán USD niêm yết Vietcombank (dùng cột **sell**) | 01/02/2020 → 17/07/2026 |
| | `fx_rates/usdvnd_market_yfinance.csv` | USDVND=X thị trường — fallback trước 02/2020 | 2004 → 17/07/2026 |
| Chứng khoán | `stock_market/vnindex_daily_2004_present.csv` | VN-Index daily (vnstock/VCI) — kênh đầu tư thay thế | 05/01/2004 → 17/07/2026 |
| Vĩ mô | `macro/macro_series_wb_gso.csv` + liên quan | CPI, GDP, cung tiền VN (GSO/WB/IMF) | 1986 → 2026 |
| Tin tức | `news/news_raw_headlines_vietnam_gold.csv` + liên quan | Tiêu đề tin tức vàng VN (Google News RSS) — đếm sự kiện/sentiment | 2010 → 11/07/2026 |
| Chính sách NHNN | `sbv_policy/policy_event_dummies.csv` | Dummy structural break: NĐ 24/2012 (độc quyền SJC), đấu thầu 2013, can thiệp NHNN 2024, NĐ 232/2025 (xóa độc quyền, hiệu lực 10/10/2025) | 2012 → 2026 |
| | `sbv_policy/deposit_rate_12m_milestones.csv` | 25 mốc lãi suất tiết kiệm 12 tháng — chi phí cơ hội giữ vàng | 2010 → 2026 |

## 3. Yêu cầu đầu vào bắt buộc cho model

| # | Yêu cầu | Nguồn | Quy ước |
|---|---|---|---|
| 1 | XAU/USD daily close | LBMA (chính) + GC=F (đối chiếu, tương quan 0,99989) | Không trùng ngày; close > 0; \|return ngày\| < 10% → flag |
| 2 | Giá SJC mua/bán daily | Raw history + PNJ update | Giá chốt cuối ngày; đơn vị triệu VND/lượng; sell > buy; spread hợp lệ theo thời kỳ: [0,01–2] triệu (2010–2021), [0,5–5] triệu (2022+) |
| 3 | Tỷ giá USD/VND | VCB sell (từ 2020) + yfinance (2004–2020) | Dùng giá **bán** VCB vì phản ánh đúng chi phí cơ hội quy đổi vàng |
| 4 | Vĩ mô thế giới | FRED + market daily + GPR | CPIAUCSL (tháng) forward-fill sang daily với **shift 1 tháng** (tránh look-ahead bias); dữ liệu Mỹ dùng giá trị t−1 cho ngày t của VN (lệch múi giờ) |
| 5 | Đặc thù Việt Nam | CPI VN, lãi suất 12T, policy dummies | Bắt buộc có dummy NĐ 232/2025 và can thiệp 2024 — thiếu sẽ khiến model học sai hành vi premium |

**Biến dẫn xuất cốt lõi** (tính ở bước chuẩn hóa):

```
gia_quy_doi (triệu/lượng) = XAU_USD × ty_gia_VCB_ban × 1,20565 / 10^6
# 1 lượng = 37,5g = 1,20565 troy oz

premium = SJC_ban − gia_quy_doi   # biến trung tâm của phần Việt Nam
```

## 4. Quy trình phân tích

### Giai đoạn 1 — Chuẩn hóa & Data Quality (tuần 1)
- Xây bảng master daily (index = ngày giao dịch VN), ghép 5 bộ dữ liệu theo quy ước ở mục 3.
- Chạy validity rules (kết quả có sẵn trong `data_quality_report.md`).
- Xử lý thiếu: forward-fill ngày lễ cho chuỗi FRED, CPI shift 1 tháng, ghi rõ mọi quyết định xử lý.

### Giai đoạn 2 — EDA (tuần 2)
- Vẽ chuỗi XAU / SJC / giá quy đổi cùng biểu đồ; premium theo thời gian với mốc structural break (6/2024, 10/2025).
- Tương quan rolling 90 ngày: XAU vs DFII10 (kỳ vọng âm mạnh), vs DXY (âm), vs VIX/GPR (dương khi khủng hoảng).
- Kiểm định Chow/t-test phân phối premium trước/sau break.
- So sánh vàng miếng SJC vs nhẫn 9999.
- Kiểm định tính dừng ADF → model trên log-return, không model trên giá gốc.
- Phân tích chu kỳ dài 2004 → nay (2008, đỉnh 2011, đáy 2015, COVID 2020, sóng 2024–2026).

### Giai đoạn 3 — Modeling (tuần 3–4)

**Tầng 1 — Giá vàng thế giới (log-return LBMA PM):**

| Model | Vai trò |
|---|---|
| Naive + Drift | Baseline bắt buộc |
| SARIMAX + exog (ΔDFII10, ΔDXY, VIX, GPR) | Model thống kê chính |
| **LightGBM/XGBoost** | Model chính đề xuất |
| LSTM/GRU | Tùy chọn so sánh |

**Tầng 2 — Premium Việt Nam:**
- `premium ~ event_dummies + lãi_suất_12T + Δtỷ_giá + trend`
- Dự báo SJC = quy đổi(dự báo XAU, tỷ giá) + dự báo premium.
- Train premium trên 2010 → nay; train Tầng 1 trên 2004 → nay.

**Validation:** walk-forward expanding window (5 fold, mỗi fold test 6 tháng); RMSE/MAE/MAPE + directional accuracy; kiểm định Diebold–Mariano so với naive. Không dùng random split.

### Giai đoạn 4 — Trả lời "Có nên mua vàng bây giờ?" (tuần 5)

Khung 3 trụ:
1. **Tín hiệu định lượng** — hướng dự báo 1–3 tháng của model tốt nhất + khoảng tin cậy.
2. **Định giá tương đối** — premium hiện tại nằm percentile nào của phân phối sau-break; real yield DFII10 đang tăng hay giảm.
3. **Kịch bản rủi ro** — 3 kịch bản Fed (giữ/hạ/tăng) → khuyến nghị có điều kiện (ví dụ: DCA nếu premium thấp và real yield giảm; đứng ngoài nếu premium cao).

**Deliverables:** notebook pipeline, notebook EDA, notebook model + bảng metrics, báo cáo cuối (kèm Data Quality + Limitations).

## 5. Cấu trúc thư mục dự án

```
Gold-Prediction/
├── data/                     # xem data/README.md để biết chi tiết từng file
│   ├── world/
│   ├── vietnam/
│   └── PROJECT_PLAN.md
├── notebooks/
│   ├── 01_pipeline.ipynb     # chuẩn hóa & ghép bảng master daily
│   ├── 02_eda.ipynb          # phân tích khám phá
│   └── 03_modeling.ipynb     # model 2 tầng + walk-forward validation
├── src/                      # module dùng chung (loader, feature engineering, model registry)
├── reports/                  # báo cáo cuối, biểu đồ, bảng metrics
└── README.md
```

## 6. Giới hạn dữ liệu cần lưu ý

- Giá SJC chỉ có từ 2010; tỷ giá VCB chính thức chỉ từ 02/2020.
- GC=F (futures) và LBMA fix lệch múi giờ giao dịch.
- Premium có structural break do thay đổi chính sách (NĐ 24/2012, NĐ 232/2025) — bắt buộc có dummy tương ứng trong model.
- Horizon dự báo ngắn (1–3 tháng) do vàng gần với random walk ở tần suất daily.
