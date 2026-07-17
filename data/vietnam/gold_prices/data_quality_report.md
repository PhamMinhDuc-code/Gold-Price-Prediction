# Data Quality Report - SJC gold (Bo 2 rules, final)
Generated: 2026-07-17 | Input: gold_raw_history_all_sources_2010_2026.csv

## Conventions (ghi vao bao cao mon hoc)
- Gia chot ngay (EOD close) = ban ghi cuoi cung cua moi (source, date) theo timestamp.
- Don vi: trieu VND/luong. 1 luong = 37.5g = 1.20556 troy oz.
- Rule spread theo thoi ky: 2010-2021 spread hop le [0.01, 2] trieu (SJC thoi ky spread hep);
  tu 2022 spread hop le [0.5, 5] trieu (dung spec goc). Ly do: kiem tra thuc nghiem cho thay
  100% ngay 2010-2021 co spread < 0.5 trieu - day la dac diem thi truong, khong phai loi du lieu.

## Source: giavang_sjc_archive
- Rows (EOD): 4,422 | Range: 2010-01-02 -> 2025-03-22
- Duplicate dates: 0
- sell > buy vi pham: 0
- Spread ngoai nguong (era-adjusted): 2 (0.05%)
- |return ngay| > 10% (flag de soat tay, khong xoa): 1

## Source: webgia_sjc_archive
- Rows (EOD): 4,856 | Range: 2010-01-02 -> 2026-07-07
- Duplicate dates: 0
- sell > buy vi pham: 0
- Spread ngoai nguong (era-adjusted): 2 (0.04%)
- |return ngay| > 10% (flag de soat tay, khong xoa): 0

## Doi chieu cheo 2 nguon (spec: sample 20-30 ngay, lech < 0.5%)
- Ngay trung nhau: 4,406 | Sample ngau nhien: 30 (seed=42)
- Dat nguong < 0.5%: 29/30 (ngay truot duy nhat: 2024-05-13, lech 0.552% - roi vao dot bien dong 5/2024, do khac thoi diem snapshot trong ngay)
- Lech trung binh tren TOAN BO 4,406 ngay trung: 0.0233% -> hai nguon nhat quan, PASS

## Bo 1 - bien luan proxy XAU
- LBMA PM fix vs GC=F (2010->nay, 3,677 ngay trung): tuong quan MUC GIA = 0.99989, tuong quan return ngay = 0.80.
- Ket luan: du an dung LBMA PM fix (world/gold_market/lbma_gold_am_pm_history.csv) lam chuoi XAU/USD chinh;
  GC=F giu vai tro doi chieu va cung cap du lieu realtime hon.

So dong can soat tay: 5 (vietnam/gold_prices/quality_flags.csv)