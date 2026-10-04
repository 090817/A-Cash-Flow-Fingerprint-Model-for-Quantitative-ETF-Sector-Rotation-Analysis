# **A Cash Flow-Fingerprint Model for ETF Sector Rotation**

> A **cash flow-driven multi-factor quantitative system** for ETF sector rotation analysis, connecting capital movement, price dynamics, and volatility stability into an interpretable composite score. Through an explicit weight constraint on cash flow factors, dual-source data redundancy, and rigorous overfitting validation, it delivers transparent, actionable sector rankings for retail and institutional investors.

---

## **📖 Project Introduction**

**FlowPrint** is a quantitative sector rotation framework designed to rank sector-specific Exchange-Traded Funds (ETFs) based on a **Cash Flow Fingerprint score**. It aims to solve common problems in traditional sector rotation strategies: over-reliance on lagging price momentum, opacity of black-box machine learning models, and the underweighting of direct capital flow signals.

The system collects daily Open-High-Low-Close-Amount (OHLCA) data from 15 sector ETFs and synthesizes five complementary factors — **Cash Flow, Momentum, Price Position, Trend Strength, and Stability** — into a single composite FlowPrint score. An explicit constraint requires the cash flow weight to exceed 30%, ensuring the model reflects actual capital movement rather than statistical artifacts. Empirical backtesting (2020–2024, out-of-sample 2023–2024) demonstrates superior risk-adjusted returns compared to equal-weight and momentum-only benchmarks.

---

## **✨ Core Features**

### **1. Factor Engine**

- **Cash Flow factor (35%)**: Combines Flow Change (40%), Price-Flow Alignment (40%), and Flow Breakout (20%) to capture the momentum of capital movement.
- **Momentum factor (25%)**: Blends 5-day log returns (70%) with relative strength (30%) to gauge short-term price direction.
- **Price Position factor (20%)**: Contextualizes the closing price within its 20-day high-low range to identify overbought/oversold conditions.
- **Trend Strength factor (15%)**: Compares 5-day and 60-day SMAs to assess trend sustainability.
- **Stability factor (5%)**: Inversely weights 20-day rolling return volatility to reward stable assets.

### **2. FlowPrint Composite Score**

- Weighted linear combination of the five factors, normalized to a 0–100 scale.
- **Hard constraint**: Cash flow weight ≥ 30% to ensure capital movement is the dominant signal.
- Default weights: Cash Flow 35%, Momentum 25%, Price Position 20%, Trend Strength 15%, Stability 5%.

### **3. Data Pipeline**

- **Dual-source redundancy**: Primary data from Baostock, fallback from AkShare.
- **Caching mechanism**: Local storage of recently fetched data, refreshed daily after market close.
- **Data validation**: Automatic checks for missing values and 3σ outliers.

### **4. Noise Filtering & Smoothing**

- **Anomaly filter**: Flags days where |return| > 15% but amount change < 10%, recalculating cash flow using a 3-day median.
- **Signal smoothing**: Applies a 3-day exponential moving average to the final composite score.

### **5. Backtesting & Validation**

- **Weekly rotation strategy**: Equal-weighted top-3 sectors, rebalanced every Friday close.
- **Benchmarks**: Equal-weight all 15 ETFs (EW) and 1-month momentum top-3.
- **Overfitting checks**: Jaccard similarity across 30/90/180-day ranking windows.
- **Formula sanity checks**: Verifies all scores fall within the 0–100 normalized range.

### **6. Output Layer**

- Ranked sector list with FlowPrint scores and component factor scores.
- Factor contribution charts illustrating the relative weight of each component.
- Consolidated validation report summarizing data quality, sanity checks, and overfitting risk.

---

## **🏗️ System Architecture**

text

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                            │
│  - OHLCA daily data for 15 sector ETFs                       │
│  - Dual-source: Baostock (primary), AkShare (fallback)       │
│  - Local cache with daily refresh                            │
│  - Data validation & cleaning (missing values, outliers)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Model Layer                            │
│  - Feature Calculator: derives 5 normalized factors          │
│  - FlowPrint Engine: applies weight constraints              │
│  - Sector Ranking Module: sorts sectors by composite score   │
│  - Noise filtering & 3-day EMA smoothing                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Validation Layer                         │
│  - Overfit check: temporal consistency (30/90/180-day)       │
│  - Formula rationality test: score range 0–100               │
│  - Data quality monitor: missing values, outliers            │
│  - Validation report generation                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Output Layer                            │
│  - Ranking table & scores                                    │
│  - Factor contribution charts                                │
│  - Validation report                                         │
│  - Web platform for rapid ETF suitability analysis           │
└─────────────────────────────────────────────────────────────┘
```

---

## **📁 Directory Structure and File Descriptions**

> The following is an inferred structure based on project functionality. **Please refer to the actual repository files.**

text

```
FlowPrint/
├── README.md                          # Project documentation
├── data/                              # Data acquisition and caching
│   ├── fetch_baostock.py              # Primary data fetcher (Baostock API)
│   ├── fetch_akshare.py               # Fallback data fetcher (AkShare API)
│   ├── cache/                         # Local cached OHLCA data
│   └── validation.py                  # Missing value & outlier checks
├── model/                             # Factor calculation and scoring
│   ├── factors.py                     # Five factor definitions (M, C, P, T, S)
│   ├── flowprint_engine.py            # Composite score with weight constraints
│   ├── noise_filter.py                # Anomaly filter and EMA smoothing
│   └── ranking.py                     # Sector ranking module
├── validation/                        # Overfitting and robustness checks
│   ├── temporal_consistency.py        # Jaccard similarity across windows
│   ├── formula_sanity.py              # Score range verification
│   └── report_generator.py            # Validation report
├── backtest/                          # Strategy backtesting
│   ├── rotation_strategy.py           # Weekly top-3 rotation
│   ├── benchmarks.py                  # EW and momentum benchmarks
│   └── performance_metrics.py         # Sharpe, drawdown, annualized return
├── output/                            # Generated outputs
│   ├── ranking_table.csv              # Ranked sector list with scores
│   ├── factor_contribution.png        # Factor contribution chart
│   └── validation_report.json         # Consolidated validation findings
├── web/                               # Web platform
│   ├── index.html                     # Main interface
│   ├── app.js                         # Score visualization and interaction
│   └── style.css                      # Styles
└── docs/                              # Documentation
    ├── methodology.md                 # Formula derivations
    └── data_dictionary.md             # Field definitions
```

### **Detailed File Descriptions**


| **File/Directory**                   | **Description**                                                                                                             |        |                                                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------- |
| `data/fetch_baostock.py`             | Primary data acquisition from Baostock API for 15 sector ETFs; retrieves daily OHLCA data.                                  |        |                                                          |
| `data/fetch_akshare.py`              | Fallback data acquisition from AkShare API to ensure availability when the primary source is down.                          |        |                                                          |
| `data/cache/`                        | Local storage of recently fetched data; refreshed daily after market close to balance timeliness and API call minimization. |        |                                                          |
| `data/validation.py`                 | Data quality checks: flags missing values and identifies 3σ outliers for review or imputation.                              |        |                                                          |
| `model/factors.py`                   | Implements the five core factors: Momentum (M), Cash Flow (C), Price Position (P), Trend Strength (T), and Stability (S).   |        |                                                          |
| `model/flowprint_engine.py`          | Computes the composite FlowPrint score with weight constraints (cash flow ≥ 30%).                                           |        |                                                          |
| `model/noise_filter.py`              | Anomaly filter for irregular trades (                                                                                       | return | > 15% with amount change < 10%) and 3-day EMA smoothing. |
| `model/ranking.py`                   | Sorts sectors by FlowPrint scores to produce a ranked list reflecting relative sector strength.                             |        |                                                          |
| `validation/temporal_consistency.py` | Compares sector rankings across 30/90/180-day windows using Jaccard similarity to detect overfitting.                       |        |                                                          |
| `validation/formula_sanity.py`       | Verifies all component and composite scores fall within the expected 0–100 normalized range.                                |        |                                                          |
| `validation/report_generator.py`     | Produces a consolidated validation report summarizing data quality, sanity checks, and overfitting risk.                    |        |                                                          |
| `backtest/rotation_strategy.py`      | Weekly rebalanced top-3 sector rotation strategy based on FlowPrint rankings.                                               |        |                                                          |
| `backtest/benchmarks.py`             | Baseline strategies: equal-weight all 15 ETFs (EW) and 1-month momentum top-3.                                              |        |                                                          |
| `backtest/performance_metrics.py`    | Computes total return, annualized return, volatility, max drawdown, Sharpe ratio, and trade count.                          |        |                                                          |
| `web/index.html`                     | Web platform entry point for rapid ETF suitability analysis.                                                                |        |                                                          |
| `web/app.js`                         | Interactive visualization of sector rankings, factor contributions, and validation reports.                                 |        |                                                          |


---

## **🧪 Data Format Examples**

### **Factor Calculation Output (excerpt)**

json

```
{
  "date": "2024-12-31",
  "sector": "Technology",
  "etf_code": "512480",
  "flowprint_score": 78.2,
  "factors": {
    "cash_flow": 85.1,
    "momentum": 72.3,
    "price_position": 65.4,
    "trend_strength": 80.1,
    "stability": 75.0
  },
  "weights": {
    "cash_flow": 0.35,
    "momentum": 0.25,
    "price_position": 0.20,
    "trend_strength": 0.15,
    "stability": 0.05
  }
}
```

### **Backtest Performance Metrics (2023–2024)**


| **Strategy**       | **Annual Return** | **Sharpe Ratio** | **Max Drawdown** |
| ------------------ | ----------------- | ---------------- | ---------------- |
| FlowPrint Rotation | 16.8%             | 0.92             | -12.4%           |
| Momentum Rotation  | 11.3%             | 0.68             | -15.8%           |
| Equal Weight (EW)  | 7.5%              | 0.50             | -14.2%           |


### **Overfitting Validation Output**

json

```
{
  "jaccard_similarity": 0.72,
  "perfect_overlap_pct": 40,
  "partial_overlap_pct": 45,
  "no_overlap_pct": 15,
  "interpretation": "High ranking consistency across 30/90/180-day windows; model is robust to overfitting."
}
```

---

## **🚀 Quick Start**

### **Requirements**

- Python 3.8+
- Pandas, NumPy, Backtrader
- Baostock and AkShare API access
- Modern browser (Chrome / Edge / Safari) for the web platform

### **Install Dependencies**

bash

```
pip install pandas numpy backtrader baostock akshare
```

### **Fetch Data**

bash

```
cd data
python fetch_baostock.py
# Fallback if primary source is down:
python fetch_akshare.py
```

### **Run Factor Calculation**

bash

```
cd model
python flowprint_engine.py
# Generates ranked sector list with FlowPrint scores
```

### **Run Backtest**

bash

```
cd backtest
python rotation_strategy.py
# Outputs performance metrics: Sharpe, drawdown, annualized return
```

### **Run Validation**

bash

```
cd validation
python temporal_consistency.py
python formula_sanity.py
python report_generator.py
```

### **Launch Web Platform**

bash

```
cd web
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

---

## **🛠️ Tech Stack**

- **Data acquisition**: Baostock, AkShare APIs
- **Data processing**: Python, Pandas, NumPy
- **Backtesting**: Backtrader
- **Visualization**: Matplotlib, Chart.js / D3.js (inferred)
- **Web platform**: HTML5, CSS3, JavaScript (ES6+)
- **Validation**: Jaccard similarity, temporal consistency analysis
- **Data storage**: CSV / JSON files, local cache

---

## **📊 Selected Sector ETFs & Benchmark**


| **Sector**       | **ETF Code** | **ETF Name**                  | **Description**            |
| ---------------- | ------------ | ----------------------------- | -------------------------- |
| Financials       | 512000       | HuaBao Securities ETF         | Brokerages, Banks          |
| Technology       | 512480       | GuoTai Semiconductor ETF      | Semiconductors, Tech       |
| Consumer Staples | 512600       | JiaShi Consumer ETF           | Food, Beverage, Household  |
| Healthcare       | 512010       | HuaBao Healthcare ETF         | Pharmaceuticals, Biotech   |
| New Energy       | 515030       | HuaXia New Energy Auto ETF    | Electric Vehicles, Battery |
| …                | …            | …                             | …                          |
| Benchmark        | 510300       | HuaTai-PineBridge CSI 300 ETF | Broad Market Index         |


---

## **📈 Key Findings**

- **FlowPrint achieved 49% higher return than momentum strategy** with 22% smaller maximum loss.
- **Optimal cash flow weight** appears to be 35–40% based on backtest results.
- **Average Jaccard similarity of 0.72** across time windows confirms model robustness to overfitting.
- **Top-ranked sectors** consistently show strong cash flow scores alongside momentum, validating the hypothesis that capital movement leads price action.

---

## **🔬 Limitations & Future Work**

### **Current Limitations**

- **Data dependency**: Limited to Chinese A-shares (2020–2024); daily frequency misses intraday signals.
- **Uniform flow treatment**: Cannot distinguish institutional vs. retail flow.
- **Static sector definitions**: Relies on pre-defined sector ETFs; thematic shifts are captured with lag.
- **Macro-regime blindness**: Does not explicitly integrate interest rate, inflation, or geopolitical shocks.

### **Future Directions**

- **Enrich cash flow factor**: Integrate order flow imbalance and mutual fund/ETF flow data.
- **Macro-regime detection layer**: Dynamic factor weight adjustment based on interest rate volatility and inflation trends.
- **Cross-market validation**: Apply FlowPrint to US, EU, and Asia-Pacific sector ETFs.
- **LLM integration**: Analyze financial news, regulatory filings, and social media sentiment to refine factors.

---

## **🤝 Contributing**

Issues and Pull Requests are welcome. Please ensure:

1. Consistent code style.
2. New features come with test data or documentation.
3. Update the corresponding file descriptions in the README.

---

## **📬 Contact**

- **Author**: Zhiqiao Li
- **Research Advisor**: Longlong Ma
- **Project Type**: Quantitative Finance Research / Machine Learning Application
- **Repository**: [https://github.com/090817/ELAK-Physicalcare-therapy](https://github.com/090817/ELAK-Physicalcare-therapy)

