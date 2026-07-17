# KẾ HOẠCH DỰ ÁN: PHÂN TÍCH & DỰ ĐOÁN GIÁ VÀNG (THẾ GIỚI + VIỆT NAM)
### Câu hỏi nghiên cứu: "Có nên mua vàng ở thời điểm hiện tại hay không?"
*Cập nhật: 17/07/2026 — dữ liệu phủ 2004 → 17/07/2026 (thế giới), 2010 → 17/07/2026 (Việt Nam)*

---

## PHẦN A — DANH MỤC DỮ LIỆU CHI TIẾT (nguồn, link, mô tả)

### A1. Nhóm THẾ GIỚI — Giá vàng & thị trường

| File | Nguồn | Link gốc | Nội dung | Phủ thời gian |
|---|---|---|---|---|
| `world/gold_market/lbma_gold_am_pm_history.csv` | LBMA (London Bullion Market Association) | https://prices.lbma.org.uk/json/gold_am.json và `gold_pm.json` | Giá vàng XAU/USD chốt phiên London sáng (AM) & chiều (PM) — **chuẩn giá vàng thế giới chính thức**, dùng làm chuỗi giá chính của dự án | 1968 → 16/07/2026 |
| `world/gold_market/market_daily_2004_present.csv` | Yahoo Finance (yfinance) | https://finance.yahoo.com/quote/GC%3DF/history | **File thị trường chính**: giá đóng cửa 8 mã — GC=F (vàng COMEX futures), SI=F (bạc), CL=F (dầu WTI), ^GSPC (S&P 500), ^VIX (chỉ số sợ hãi), DX-Y.NYB (chỉ số USD), GLD (ETF vàng), USDVND=X | 2004 → 17/07/2026 |
| `world/gold_market/gld_shares_outstanding.csv` | SEC EDGAR (báo cáo SPDR Gold Trust) | https://www.sec.gov/cgi-bin/browse-edgar (CIK: SPDR Gold Trust) | Số chứng chỉ quỹ GLD đang lưu hành — đo dòng tiền vào/ra ETF vàng (tâm lý nhà đầu tư tổ chức) | 2010 → 07/2026 (theo tháng) |
| `world/gold_market/global_market_series_yfinance_fred.csv` | Yahoo Finance + FRED | như trên | Bản cũ của pipeline (2011→10/07/2026) — đã được `market_daily_2004_present.csv` thay thế, giữ để đối chiếu | 2011 → 10/07/2026 |
| `world/gold_market/etf_proxy_gld.csv`, `futures_basis_gc_f.csv`, `lbma_gold_proxy_gc_f.csv` | Dẫn xuất từ Yahoo Finance | — | Các chuỗi phụ trợ: proxy ETF, basis futures | 2010 → 10/07/2026 |
| `world/gold_market/gspc_2010_2011_supplement.csv` | Yahoo Finance | https://finance.yahoo.com/quote/%5EGSPC/history | Bổ sung S&P 500 đoạn 2010 → 07/2011 | 2010–2011 |

### A2. Nhóm THẾ GIỚI — Vĩ mô Mỹ (FRED)

| File | Nguồn | Link gốc | Nội dung | Phủ thời gian |
|---|---|---|---|---|
| `world/macro_fred/fred_series_full_history.csv` | FRED (Fed St. Louis) | https://fred.stlouisfed.org/series/DGS10 (thay ID tương ứng) | **File FRED hợp nhất, 14 series**: DGS10 (lợi suất 10Y, 1962→), DFII10 (lợi suất thực TIPS 10Y — *biến quan trọng nhất với vàng*, 2003→), DFF (lãi suất Fed, 1954→), CPIAUCSL (CPI Mỹ tháng, 1947→), DTWEXBGS (USD index rộng, 2006→), VIXCLS (1990→), DCOILWTICO (dầu WTI, 1986→); T10YIE, T5YIE (kỳ vọng lạm phát), NFCI, STLFSI2 (stress tài chính), AAA10Y, BAA10Y (spread tín dụng), M2SL (cung tiền) từ 2010 | 1947 → 14/07/2026 |
| `world/macro_fred/t10yie_computed_2003_present.csv` | Tự tính (DGS10 − DFII10) | — | Kỳ vọng lạm phát 10Y (breakeven) kéo dài về 2003 | 2003 → 14/07/2026 |
| `world/macro_fred/DFF.csv`, `CPIAUCSL.csv` | FRED (tải thủ công) | https://fred.stlouisfed.org/series/DFF , /CPIAUCSL | File gốc lãi suất Fed & CPI Mỹ | 1954/1947 → 07/2026 |
| `world/macro_fred/macro_enhanced_fred_expanded.csv` | FRED (pipeline cũ) | — | Bản 12 series từ 2010 (đã gộp vào file hợp nhất) | 2010 → 10/07/2026 |

⚠️ Ghi chú: STLFSI2 ngừng cập nhật 01/2022 (FRED thay bằng STLFSI4) — loại khỏi model hoặc tải STLFSI4 thủ công.

### A3. Nhóm THẾ GIỚI — Rủi ro địa chính trị

| File | Nguồn | Link gốc | Nội dung | Phủ thời gian |
|---|---|---|---|---|
| `world/geopolitical_risk/gpr_daily_geopolitical_risk.csv` | Caldara & Iacoviello (Fed Board) | https://www.matteoiacoviello.com/gpr.htm | Chỉ số rủi ro địa chính trị GPR theo ngày (đếm tin tức chiến tranh/khủng bố trên báo chí) — vàng thường tăng khi GPR tăng đột biến | 1985 → 13/07/2026 |
| `world/geopolitical_risk/gpr_monthly_geopolitical_risk.csv` | như trên | như trên | Bản theo tháng (về tận 1900) | 1900 → 2026 |

### A4. Nhóm VIỆT NAM — Giá vàng trong nước

| File | Nguồn | Link gốc | Nội dung | Phủ thời gian |
|---|---|---|---|---|
| `vietnam/gold_prices/gold_raw_history_all_sources_2010_2026.csv` | Crawl từ giavang.org + webgia.com + SJC + PNJ | https://giavang.org/ , https://webgia.com/gia-vang/sjc/ , https://sjc.com.vn , https://www.pnj.com.vn/site/gia-vang | **File giá vàng VN chính**: giá mua/bán vàng miếng SJC (và PNJ) mọi lần đổi giá trong ngày, 2 archive độc lập để đối chiếu chéo | 02/01/2010 → 07/07/2026 |
| `vietnam/gold_prices/gold_recent_update_pnj_sjc.csv` | API chính thức PNJ | https://edge-api.pnj.io/ecom-frontend/v1/get-gold-price-history?date=YYYYMMDD | Cập nhật mới nhất: SJC miếng + nhẫn PNJ 9999, giá chốt cuối ngày (API có lịch sử về ~2015 — nguồn đối chiếu thứ 3) | 01/07 → 17/07/2026 |
| `vietnam/gold_prices/gold_quotes_sjc_historical.csv` | SJC chính thức | https://sjc.com.vn/bieu-do-gia-vang | Giá SJC theo chi nhánh (HCM, HN...) | 2011 → 11/07/2026 |
| `vietnam/gold_prices/canonical_gold_products_daily.csv` | Chuẩn hóa từ raw | — | Giá daily theo từng sản phẩm: miếng SJC 1L, nhẫn 9999... (dùng để so miếng vs nhẫn) | 2010 → 11/07/2026 |
| `vietnam/gold_prices/data_quality_report.md` + `quality_flags.csv` | Tự sinh (kiểm định) | — | Báo cáo chất lượng: 0 vi phạm sell>buy, đối chiếu chéo 2 nguồn lệch TB 0,023%, 5 dòng flag | — |

### A5. Nhóm VIỆT NAM — Tỷ giá, chứng khoán, vĩ mô, chính sách

| File | Nguồn | Link gốc | Nội dung | Phủ thời gian |
|---|---|---|---|---|
| `vietnam/fx_rates/vcb_usd_daily.csv` | Vietcombank (API chính thức) | https://www.vietcombank.com.vn/api/exchangerates?date=YYYY-MM-DD | Tỷ giá USD niêm yết VCB: mua tiền mặt / mua chuyển khoản / **bán ra** (cột dùng để quy đổi giá vàng) | 01/02/2020 → 17/07/2026 |
| `vietnam/fx_rates/usdvnd_market_yfinance.csv` | Yahoo Finance | https://finance.yahoo.com/quote/USDVND%3DX | Tỷ giá USD/VND thị trường — fallback cho giai đoạn trước 02/2020 | 2004 → 17/07/2026 |
| `vietnam/stock_market/vnindex_daily_2004_present.csv` | vnstock (API VCI) | https://github.com/thinh-vu/vnstock | VN-Index đóng cửa daily — kênh đầu tư thay thế vàng, đo risk-on/risk-off nội địa | 05/01/2004 → 17/07/2026 |
| `vietnam/macro/macro_series_wb_gso.csv` + 2 file | GSO / World Bank / IMF | https://www.gso.gov.vn , https://data.worldbank.org | CPI Việt Nam, GDP, cung tiền... (tháng/năm) | 1986 → 2026 |
| `vietnam/news/news_raw_headlines_vietnam_gold.csv` + 2 file | Google News RSS | https://news.google.com/rss (query "giá vàng") | Tiêu đề tin tức vàng VN — dùng đếm sự kiện/sentiment | 2010 → 11/07/2026 |
| `vietnam/sbv_policy/policy_event_dummies.csv` | Tự lập từ văn bản pháp luật + thông cáo NHNN | NĐ 24/2012: https://vanban.chinhphu.vn ; NĐ 232/2025: https://xaydungchinhsach.chinhphu.vn ; sbv.gov.vn | **Bảng dummy structural break**: NĐ 24/2012 (độc quyền SJC), đấu thầu 2013, đấu thầu + bán bình ổn NHNN 2024, NĐ 232/2025 hiệu lực 10/10/2025 (xóa độc quyền) | 2012 → 2026 |
| `vietnam/sbv_policy/deposit_rate_12m_milestones.csv` | Tự lập từ thông cáo NHNN/VCB/báo chí | https://www.sbv.gov.vn , https://www.vietcombank.com.vn/vi-VN/KHCN/Cong-cu-Tien-ich/Lai-suat | 25 mốc thay đổi lãi suất tiết kiệm 12 tháng (trần 14% 2011 → đáy 4,6% 2024) — chi phí cơ hội của việc giữ vàng | 2010 → 2026 |
| `vietnam/sbv_policy/sbv_gold_policy_events.csv`, `retail_deposit_rates.csv` | sbv.gov.vn crawl | https://www.sbv.gov.vn | Sự kiện đấu thầu vàng 16/07/2026; lãi suất niêm yết hiện tại 15 ngân hàng | 07/2026 |

---

## PHẦN B — YÊU CẦU ĐẦU VÀO (5 bộ dữ liệu bắt buộc cho model)

| Bộ | Yêu cầu | File đáp ứng | Quy ước bắt buộc |
|---|---|---|---|
| 1 | XAU/USD daily close | `lbma_gold_am_pm_history.csv` (chính) + GC=F (đối chiếu, tương quan 0,99989) | Không trùng ngày; close > 0; \|return ngày\| < 10% nếu vượt thì flag |
| 2 | Vàng SJC mua/bán daily | `gold_raw_history_all_sources...csv` + `gold_recent_update_pnj_sjc.csv` | **Giá chốt cuối ngày** (SJC đổi giá nhiều lần/ngày); đơn vị triệu VND/lượng; sell > buy mọi dòng; spread hợp lệ theo thời kỳ: [0,01–2] triệu (2010–2021), [0,5–5] triệu (2022+); đối chiếu chéo ≥2 nguồn — ĐÃ PASS (lệch TB 0,023%) |
| 3 | Tỷ giá USD/VND | `vcb_usd_daily.csv` (cột `sell`, từ 02/2020) + `usdvnd_market_yfinance.csv` (2004–2020) | Dùng **giá bán VCB** vì đó là giá người dân/DN thực mua USD — đúng chi phí cơ hội khi quy đổi vàng; tỷ giá trung tâm NHNN chỉ là tham chiếu, lệch xa giao dịch thực |
| 4 | Vĩ mô thế giới | `fred_series_full_history.csv` + `market_daily_2004_present.csv` + GPR | CPIAUCSL là dữ liệu THÁNG → forward-fill sang daily **với shift 1 tháng** (CPI công bố trễ, tránh look-ahead bias); dữ liệu Mỹ dùng giá trị t−1 cho ngày t của VN (lệch múi giờ) |
| 5 | Đặc thù VN | CPI VN (`macro/`), lãi suất 12T (`deposit_rate_12m_milestones.csv` — fill từ mốc), `policy_event_dummies.csv` | **Bắt buộc có dummy NĐ 232/2025 và can thiệp 2024** — thiếu chúng model sẽ học sai hành vi premium do structural break |

**Biến dẫn xuất cốt lõi** (tính trong bước chuẩn hóa):
- `gia_quy_doi (triệu/lượng) = XAU_USD × ty_gia_VCB_ban × 1,20556 / 10^6` (1 lượng = 37,5g = 1,20556 troy oz)
- `premium = SJC_ban − gia_quy_doi` — "nhân vật chính" của phần Việt Nam

---

## PHẦN C — CÁC BƯỚC PHÂN TÍCH

### Giai đoạn 1 — Chuẩn hóa & Data Quality (tuần 1)
1. Xây **bảng master daily** (index = ngày giao dịch VN): ghép Bộ 1–5 theo quy ước ở Phần B.
2. Chạy validity rules (đã có sẵn kết quả trong `data_quality_report.md` — đưa vào phụ lục báo cáo).
3. Xử lý thiếu: forward-fill ngày lễ cho chuỗi FRED; CPI shift 1 tháng; nêu rõ mọi quyết định trong báo cáo.

### Giai đoạn 2 — EDA (tuần 2)
1. Chuỗi XAU, SJC, giá quy đổi cùng biểu đồ; **premium theo thời gian** với 2 đường dọc đánh dấu structural break (can thiệp 6/2024, NĐ 232 10/2025).
2. Tương quan rolling 90 ngày: XAU vs DFII10 (kỳ vọng âm mạnh), vs DXY (âm), vs VIX/GPR (dương khi khủng hoảng).
3. Phân phối premium trước/sau từng break (kiểm định Chow / t-test) — chứng minh sự cần thiết của dummy.
4. So sánh vàng miếng SJC vs nhẫn 9999 (insight riêng VN: nhẫn ít bị bóp méo bởi độc quyền hơn).
5. Kiểm định tính dừng ADF → model trên log-return, không model trên giá gốc.
6. Phân tích chu kỳ dài 2004→nay: khủng hoảng 2008, đỉnh 2011, đáy 2015, COVID 2020, sóng 2024–2026.

### Giai đoạn 3 — Modeling (tuần 3–4): kiến trúc 2 tầng

**Tầng 1 — Giá vàng thế giới (log-return LBMA PM):**
| Model | Vai trò |
|---|---|
| Naive + Drift | Baseline bắt buộc (vàng ≈ random walk; model không thắng naive = vô nghĩa) |
| SARIMAX + exog (ΔDFII10, ΔDXY, VIX, GPR) | Model thống kê chính — giải thích được hệ số |
| **LightGBM/XGBoost** (lag return 1–21, MA ratio, RSI, ΔDFII10, ΔDFF, VIX, dầu, breakeven, GPR) | Đề xuất làm **model chính** — thường tốt nhất với dữ liệu daily dạng bảng |
| LSTM/GRU (tùy chọn) | Để so sánh nếu còn thời gian; ~5.600 điểm daily thường không đủ để thắng LightGBM |

**Tầng 2 — Premium Việt Nam (chuỗi riêng, mean-reverting có break):**
- `premium ~ event_dummies + lãi_suất_12T + Δtỷ_giá + trend` (OLS/GLS hoặc LightGBM nhỏ)
- Dự báo SJC = quy đổi(dự báo XAU, tỷ giá) + dự báo premium.
- Train premium trên 2010→nay; train tầng 1 trên 2004→nay.

**Validation:** walk-forward expanding window (5 fold, mỗi fold test 6 tháng); metrics RMSE/MAE/MAPE + **directional accuracy** (quan trọng nhất cho quyết định mua/bán); kiểm định Diebold–Mariano từng model vs naive. Tuyệt đối không dùng random split.

### Giai đoạn 4 — Trả lời "Có nên mua vàng bây giờ?" (tuần 5)
Khung 3 trụ (không trả lời bằng 1 con số dự báo):
1. **Tín hiệu định lượng**: hướng dự báo 1–3 tháng của model tốt nhất + khoảng tin cậy.
2. **Định giá tương đối**: (a) premium SJC hiện tại nằm ở percentile nào của phân phối sau-break 10/2025? — premium cao = mua vàng VN "đắt" so với thế giới; (b) real yield DFII10 đang trong regime tăng hay giảm — môi trường nghịch/thuận cho vàng.
3. **Kịch bản rủi ro**: 3 kịch bản Fed (giữ/hạ/tăng lãi suất) → khuyến nghị **có điều kiện**, ví dụ: "Mua từng phần (DCA) nếu premium < X triệu và real yield đang giảm; đứng ngoài nếu premium > Y triệu bất kể dự báo".

**Deliverables:** notebook pipeline, notebook EDA, notebook model + bảng metrics, báo cáo cuối (kèm mục Data Quality + Limitations: SJC chỉ từ 2010, VCB từ 2020, GC=F/LBMA lệch giờ, premium có break, horizon dự báo ngắn).
