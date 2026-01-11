# NZX Website Data Sources Plan

## Overview

This document identifies potential data sources from the NZX website that could be useful for ML purposes. 

**Critical requirement:** Data must be **time series** (historical data exists) since the scraper runs on-demand, not 24/7.

**Key constraint:** NZ has no centralized filing system like SEC EDGAR. NZX announcements (2017+) is the best single source.

---

## Potentially Useful Sources (Historical Data Available)

### 1. Company Announcements ✅
**URL:** `https://www.nzx.com/markets/NZSX/announcements`  
**Per-company:** `https://www.nzx.com/companies/{TICKER}/announcements`

**Historical:** Yes - announcements are timestamped and paginated back in time.

**Data Available:**
- Announcement type (MKTUPDTE, GENERAL, SHINTR, SECISSUE, ADMIN)
- Ticker symbol
- Announcement title
- Timestamp
- Price sensitive flag (P)

**ML Value:**
- Announcement frequency per ticker
- Types of announcements (sentiment proxy)
- NTA (Net Tangible Asset) updates for ETFs/funds
- Substantial holder interest (SHINTR) changes
- Security issues/buybacks

**Notes:** Announcements page shows "last 200" - may need pagination or API exploration.

---

### 2. Historical Dividends ❌
**URL:** `https://www.nzx.com/instruments/{TICKER}/dividends`  
**Market-wide:** `https://www.nzx.com/markets/NZSX/dividends`

**Historical:** Only ~2 years available - **insufficient for ML**.

**Notes:** Yahoo Finance dividend data (already in pipeline) has better historical coverage. Not worth implementing NZX dividend scraping.

---

### 3. Instrument Fundamentals ✅
**URL:** `https://www.nzx.com/instruments/{TICKER}`

**Historical:** Partial - current snapshot only, but can build history via scraping over time.

**Data Available:**
- P/E ratio
- EPS
- NTA (Net Tangible Assets)
- Gross Dividend Yield
- Securities Issued
- Market Capitalization
- ISIN

**ML Value:**
- Valuation metrics
- Cross-sectional fundamentals
- Shares outstanding for per-share calculations

**Notes:** This supplements Yahoo Finance data but comes directly from NZX.

---

### 4. Upcoming Results Calendar ✅
**URL:** `https://www.nzx.com/markets/NZSX/upcoming_results`

**Historical:** No direct history, but can build history via regular scraping.

**Data Available:**
- Ticker
- Result type (HALFYR, FLLYR)
- Expected announcement date

**ML Value:**
- Earnings announcement timing (event studies)
- Days until next earnings (volatility predictor)
- Reporting schedule patterns

---

### 5. NZX Shareholder Metrics (Monthly) ✅
**URL:** Via announcements - `NZX Shareholder Metrics - {Month} {Year}`

**Historical:** Yes - published monthly as announcements.

**Data Available:**
- Total shareholder accounts
- Market-wide ownership statistics
- Retail vs institutional breakdown (if available)

**ML Value:**
- Market sentiment proxy
- Retail participation trends
- Market breadth indicators

---

### 6. Fund NAV (Net Asset Value) Updates ✅
**URL:** Via announcements (daily for many ETFs/funds)

**Historical:** Yes - timestamped announcements contain NTA values.

**Data Available:**
- Daily NTA per unit for ETFs/funds
- Format: `{TICKER} NTA {date} ${value}`

**ML Value:**
- Premium/discount to NAV for ETFs
- Cross-market valuation (for international ETFs)
- Fund performance tracking

**Relevant Tickers:** ASP, AGG, APA, ASD, ASF, ASR, AUE, BOT, BTC, CO2, DIV, EMF, EMG, EUF, EUG, ESG, FNZ, GBF, GGB, GLD, GPR, INF, JPN, LIV, MDZ, MZY, NGB, NPF, NZB, NZC, NZG, NZT, OZY, TNZ, TWF, TWH, USA, USF, USG, USH, USM, USS, UST, USV, etc.

---

### 7. Company Analysis Summaries ✅
**URL:** `https://www.nzx.com/companies/{TICKER}/analysis`

**Historical:** Partial - summaries may be updated periodically.

**Data Available:**
- Business overview
- Recent performance summary
- Key financial highlights (extracted from results)

**ML Value:**
- Text for NLP/sentiment analysis
- Structured performance data

---

## Not Suitable (No Historical Data / Non-Time Series)

### Upcoming Meetings ❌
**URL:** `https://www.nzx.com/markets/NZSX/upcoming_meetings`  
**Issue:** Forward-looking only, no historical record of past meetings.

### Upcoming Listings ❌
**URL:** `https://www.nzx.com/markets/NZSX/upcoming_listings`  
**Issue:** Point-in-time, empties after listing occurs.

### Price Charts ❌
**URL:** `https://www.nzx.com/instruments/{TICKER}` (TradingView charts)  
**Issue:** Charts are rendered client-side via TradingView - no scrapable historical data. Use Yahoo Finance instead.

### Real-Time Data Feed ❌
**URL:** Requires NZX license  
**Issue:** Paid product, requires licensing agreement.

---

## External Related Sources (Time Series Available)

### Global Dairy Trade (GDT) ✅
**URL:** `https://www.globaldairytrade.info/en/product-results/`

**Historical:** Yes - 10+ years of auction data available.

**Data Available:**
- GDT Price Index
- Individual product prices (WMP, SMP, Butter, AMF, Cheddar, etc.)
- Auction dates (bi-weekly)
- Volume sold
- Number of bidders

**ML Value:**
- Commodity price trends (NZ dairy exports)
- Relevant for: Fonterra (FCG), A2 Milk (ATM), Synlait (SML)

**Notes:** Already partially implemented in `GlobalAquacultureProduction.java` - could expand.

---

### NZ Milk Production (via NZX Dairy Insight) ✅
**URL:** `https://www.nzx.com/products/dairy-insight/nz-milk-production`

**Historical:** Yes - monthly reports with historical data.

**Data Available:**
- Monthly milk production volumes
- Regional breakdown
- Season-to-date totals

**ML Value:**
- Agricultural sector health
- Dairy company fundamentals proxy

**Notes:** Published as PDF - would need PDF parsing.

---

## Implementation Priority

| Priority | Source | Effort | ML Value | Notes |
|----------|--------|--------|----------|-------|
| 1 | **Announcements + PDF Parsing** | High | Very High | Richest fundamental data, 2017+ |
| 2 | Fund NAV Updates | Low | Medium | Via announcements |
| 3 | Instrument Fundamentals | Low | Medium | Snapshot data |
| 4 | GDT Auction Data | Medium | High | Dairy stocks only |
| 5 | Shareholder Metrics | Low | Low | Monthly aggregates |
| 6 | Upcoming Results | Low | Medium | Event timing |

**Removed:** Historical Dividends (insufficient depth - use Yahoo Finance)

---

## URL Patterns Summary

```
# Market-wide
https://www.nzx.com/markets/NZSX/announcements
https://www.nzx.com/markets/NZSX/dividends
https://www.nzx.com/markets/NZSX/upcoming_results
https://www.nzx.com/markets/NZSX/upcoming_meetings

# Per-company
https://www.nzx.com/companies/{TICKER}
https://www.nzx.com/companies/{TICKER}/announcements
https://www.nzx.com/companies/{TICKER}/documents
https://www.nzx.com/companies/{TICKER}/analysis

# Per-instrument
https://www.nzx.com/instruments/{TICKER}
https://www.nzx.com/instruments/{TICKER}/dividends

# External
https://www.globaldairytrade.info/en/product-results/
https://www.globaldairytrade.info/en/product-results/{product}/
```

---

## Key Findings

### No Centralized NZ Filing System

Unlike the US (SEC EDGAR) or UK (Companies House), **New Zealand has no centralized repository** for listed company annual reports and financials.

| Source Investigated | Result |
|---------------------|--------|
| **NZX Announcements** | ✅ Best option - 2017 to present (~8 years) |
| **Disclose Register** | ❌ For managed funds (KiwiSaver, ETFs), not listed equities |
| **Companies Office** | ❌ Registration info only, no financial filings |
| **FMA Registers** | ❌ Regulatory status only |
| **Company IR Sites** | ⚠️ Fragmented, variable depth (some 10+ years) |

**Implication:** NZX announcements (2017+) is the best single source for historical documents. Company IR sites may have deeper history but require per-company scraping.

### Dividend Data Depth

Historical dividends only contain ~2 years of data on NZX website - **insufficient for ML**. Yahoo Finance dividend data (already in pipeline) has better coverage.

---

## Document Parsing Strategy

Annual reports and financial documents are the richest source of fundamental data. These are published as PDFs via NZX announcements.

### Approach: Local LLM Extraction

Given existing GPU hardware for ML, local LLM parsing is cost-effective and avoids API dependencies.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT PARSING PIPELINE                                │
└────────────────────────────────────────────────────────────────────────────┘

 NZX Announcements
       │
       ▼
┌──────────────────┐
│ PDF Collection   │   Download annual reports, interim reports
│ (Java or Python) │   Store: data/documents/{ticker}/{date}_{type}.pdf
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Text Extraction  │   Extract text from PDF pages
│ (Python)         │   Options: pdfplumber, PyMuPDF, PDFBox
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLM Extraction   │   Structured data extraction via local model
│ (GPU)            │   Options: Ollama, vLLM, llama.cpp, transformers
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Validation       │   Schema validation, sanity checks
│ (Pydantic etc.)  │   Output: JSON with extracted metrics
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Integration      │   Merge into data_long.csv or separate store
└──────────────────┘
```

### PDF Text Extraction Options

| Library | Language | Strengths |
|---------|----------|-----------|
| **pdfplumber** | Python | Tables, clean text, widely used |
| **PyMuPDF (fitz)** | Python | Fast, good layout preservation |
| **PDFBox** | Java | Native to existing pipeline |
| **pdf2image + OCR** | Python | For scanned documents (rarely needed for NZX) |

### Local LLM Options

| Model Family | Sizes | Notes |
|--------------|-------|-------|
| **Qwen2.5** | 7B, 14B, 32B, 72B | Strong structured output, good at JSON |
| **Llama 3.1/3.2** | 8B, 70B | General purpose, well-supported |
| **Mistral** | 7B, 8x7B | Fast inference, good quality |
| **Phi-3/3.5** | 3.8B, 14B | Smaller, efficient |

### Inference Server Options

| Tool | Strengths |
|------|-----------|
| **Ollama** | Simplest setup, native JSON mode, good for dev |
| **vLLM** | Fastest batched inference, OpenAI-compatible API |
| **llama.cpp** | CPU offload, GGUF quantization, low VRAM |
| **transformers** | Direct HuggingFace integration, most flexible |
| **TGI** | Production-grade, Docker-friendly |

### Target Data to Extract

From annual/interim reports:

| Field | Description |
|-------|-------------|
| Revenue | Total revenue/sales |
| Net Profit | NPAT |
| Total Assets | Balance sheet |
| Total Equity | Shareholders' equity |
| EPS | Earnings per share |
| DPS | Dividend per share |
| Operating Cash Flow | Cash flow statement |
| Debt | Total borrowings |
| Fiscal Year End | Reporting period |

### Estimated Scale

| Metric | Estimate |
|--------|----------|
| NZX Tickers | ~139 |
| Years (2017-2025) | ~8 |
| Reports per ticker/year | ~2 (annual + interim) |
| **Total Documents** | ~2,200 |
| Processing time (7B model) | ~15-20 hours |
| Processing time (14B model) | ~25-35 hours |

---

## Implementation Notes

1. NZX data is delayed by 20 minutes on the public website
2. Some data requires corporate actions subscription for full history
3. Many pages use JavaScript rendering - may need Selenium/Playwright for some content
4. Rate limiting should be implemented to avoid IP blocks
5. Terms of use should be reviewed before large-scale scraping
6. NZX announcements only go back to 2017 - earlier years return "no matching records"
7. Document parsing should be a separate Python service, not part of Java ingestion
