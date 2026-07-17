# Data Directory

Datasets are organized by origin: `vietnam/` (domestic sources) and `world/` (international sources).

## vietnam/ — Vietnamese data

- `gold_prices/` — domestic gold quotes crawled from SJC, WebGia, PNJ
  - `gold_raw_history_all_sources_2010_2026.csv` — full raw crawl history (main file)
  - `backups/` — intermediate snapshots (`.pre_final`, `.pre_pnj_fix`, `.pre_repair`)
  - `gold_quotes_sjc_historical.csv` — official SJC history by branch (sjc.com.vn)
  - `domestic_gold_quotes.csv`, `domestic_target/` — audited/normalized SJC quotes
  - `canonical_gold_products_daily.csv` — per-product daily consolidation
  - `data_quality_report.md`, `quality_flags.csv` — Bộ 2 validity checks (sell>buy, era-adjusted spread bounds, return flags, cross-source verification 2 archives)
  - `gold_recent_update_pnj_sjc.csv` — SJC bar + PNJ ring EOD quotes 2026-07-01 → 2026-07-17 crawled from edge-api.pnj.io (`get-gold-price-history?date=YYYYMMDD`; API has history back to ~2015 — usable as a 3rd cross-check source)
- `fx_rates/` — USD/VND exchange rates
  - `vcb_usd_daily.csv` — Vietcombank posted USD rates (buy cash / buy transfer / **sell**), crawled from vietcombank.com.vn API; available 2020-01-28 → present (API limit)
  - `usdvnd_market_yfinance.csv` — USDVND=X market rate 2004 → present (fallback for pre-2020 period)
- `macro/` — GSO / World Bank / IMF macro series (`macro_series_wb_gso.csv`, `vn_macro_forecasting.csv`, `vn_macro_asof_panel.csv`)
- `stock_market/` — VN-Index and market series via vnstock/VCI
  - `vnindex_daily_2004_present.csv` — VNINDEX daily close 2004-01-05 → present (vnstock/VCI)
- `news/` — Vietnamese gold news headlines and events from Google News RSS
- `sbv_policy/` — State Bank of Vietnam gold policy events, policy/deposit rates, and `source_discovery/` crawl metadata
  - `deposit_rate_12m_milestones.csv` — hand-compiled 12-month VND deposit rate milestones 2010→present (VCB/big-4 reference; `confidence` column marks approximate values — verify against SBV/VCB announcements before final report)
  - `policy_event_dummies.csv` — hand-built structural-break dummy table: Nghị định 24/2012 (SJC monopoly), 2013 auctions, NHNN 2024 auctions + direct sales, Nghị định 232/2025 (monopoly removed, effective 2025-10-10)

## world/ — International data

- `gold_market/` — Yahoo Finance (GC=F futures, GLD ETF), LBMA fixes, SEC EDGAR GLD shares outstanding
  - `lbma_gold_am_pm_history.csv` — **full LBMA AM/PM fix history 1968 → present** (prices.lbma.org.uk) — primary XAU/USD series; level correlation with GC=F = 0.9999
  - `gspc_2010_2011_supplement.csv` — ^GSPC closes 2010-01 → 2011-07-05 (fills the gap before `global_market_series_yfinance_fred.csv`, which already contains ^GSPC, USDVND=X, DX-Y.NYB, ^VIX, GC=F, SI=F, CL=F from 2011-07-06)
  - `market_daily_2004_present.csv` — **primary market file**: GC=F, SI=F, CL=F, ^GSPC, ^VIX, DX-Y.NYB, GLD, USDVND=X daily closes 2004 → present (yfinance, long format) — supersedes the two files above for modeling
- `macro_fred/` — FRED macro series (US yields, CPI, ...; incl. `DFF.csv`, `CPIAUCSL.csv`)
  - `fred_series_full_history.csv` — 14 series combined, long format (date, series_id, value, coverage). Full history for DGS10 (1962→), DFF (1954→), CPIAUCSL (1947→), DCOILWTICO (1986→), VIXCLS (1990→), DFII10 (2003→), DTWEXBGS (2006→); T10YIE/T5YIE/NFCI/STLFSI2/AAA10Y/BAA10Y/M2SL only from 2010 (FRED blocks automated download; sourced from pipeline file). NOTE: STLFSI2 discontinued 2022-01 (successor: STLFSI4)
  - `t10yie_computed_2003_present.csv` — breakeven inflation computed as DGS10 − DFII10, extends T10YIE back to 2003
- `geopolitical_risk/` — GPR index, daily and monthly (Caldara & Iacoviello, matteoiacoviello.com/gpr.htm)

## Other

- `temp/` — scratch scripts (not datasets)
