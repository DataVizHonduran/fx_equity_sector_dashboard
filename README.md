# FX Fair Value Dashboard - Equity Sector Model

Interactive dashboard for foreign exchange fair value analysis using equity sector performance as predictive signals.

## Live Dashboard

🔗 **[View Dashboard](https://datavizhonduran.github.io/fx_equity_sector_dashboard/fx_equity_sector_fair_value.html)**

## Overview

This dashboard estimates "fair value" for major currency pairs by modeling FX rates as a function of equity sector relative performance. The core insight: equity sector rotation often anticipates or reflects the same macro factors that drive currency movements.

### Methodology

1. **Predictors**: Equity sector ETF performance relative to SPY (S&P 500)
   - Energy (XLE), Utilities (XLU), Industrials (XLI), Healthcare (XLV)
   - Consumer Discretionary (XLY), Staples (XLP), Technology (XLK)
   - Financials (XLF), Materials (XLB), Retail (XRT)

2. **Model**: LASSO Regression (L1 regularization, α=0.9)
   - Automatically selects which sectors matter for each currency
   - Reduces overfitting through coefficient shrinkage
   - 70/30 train-test split for validation

3. **Output**: Fair value estimates and residuals
   - **Positive residual** = Currency undervalued vs. equity signals
   - **Negative residual** = Currency overvalued vs. equity signals

### Example: Why This Works

**USD/BRL (Brazilian Real)**
- Brazil is commodity-heavy → Energy (XLE) and Materials (XLB) often have high coefficients
- When commodity sectors outperform, BRL typically strengthens
- Model captures this relationship and flags deviations

**EUR/USD**
- Europe has large industrial/automotive base → Industrials (XLI) may matter
- Financials (XLF) reflect rate differentials and capital flows
- Technology (XLK) can signal risk appetite shifts

## Features

- **Main Dashboard**: Overview of all currencies with residuals ranked
- **Individual Currency Pages**: Detailed analysis with:
  - Actual FX rate vs. model prediction
  - Residual time series (deviation from fair value)
  - Top sector predictors with coefficients
  - Model performance metrics (R², MSE)

- **Automated Updates**: Runs daily at 1:00 PM UTC via GitHub Actions

## Currencies Covered

The dashboard analyzes all currencies available in the [EMFX Risk Diffusion](https://github.com/DataVizHonduran/EMFX_risk_diffusion) dataset:

**Major Currencies**: EUR, AUD, CAD, GBP, JPY, SEK, NOK, NZD, CHF

**Emerging Markets**:
- Latin America: MXN, CLP, BRL, COP, PEN
- Asia: KRW, IDR, INR, THB, PHP, SGD
- EMEA: PLN, HUF, CZK, ZAR, TRY

## Data Sources

1. **Equity Data**: [Stooq.com](https://stooq.com) via pandas-datareader
   - Daily close prices for sector ETFs
   - 10-year historical coverage

2. **FX Data**: [EMFX Risk Diffusion CSV](https://github.com/DataVizHonduran/EMFX_risk_diffusion)
   - Automatically updated daily
   - Raw USD/XXX exchange rates from Alpha Vantage

## Usage

### Viewing the Dashboard

Simply visit the [live dashboard](https://datavizhonduran.github.io/fx_equity_sector_dashboard/fx_equity_sector_fair_value.html) and click on any currency to see detailed analysis.

### Running Locally

```bash
# Clone the repository
git clone https://github.com/DataVizHonduran/fx_equity_sector_dashboard.git
cd fx_equity_sector_dashboard

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python generate_fx_equity_dashboard.py

# Open the main dashboard
open fx_equity_sector_fair_value.html
```

### Interpreting Results

**High R² (>0.7)**: Strong relationship between equity sectors and FX rate
- Model predictions are reliable
- Residuals signal meaningful mispricings

**Low R² (<0.5)**: Weak relationship
- Other factors dominate (rates, politics, etc.)
- Use residuals cautiously

**Large Positive Residual**: Currency may be undervalued
- Actual rate is higher than equity signals suggest
- Potential mean reversion opportunity (long FX)

**Large Negative Residual**: Currency may be overvalued
- Actual rate is lower than equity signals suggest
- Potential mean reversion opportunity (short FX)

## Model Limitations

1. **Correlation ≠ Causation**: Sectors and FX may both respond to a third factor
2. **Regime Changes**: Relationships can break during crises or structural shifts
3. **Lagged Effects**: Equity signals may lead/lag FX by variable periods
4. **US Equity Bias**: All sectors are US-listed ETFs (may not capture local dynamics)

## Technical Details

- **Language**: Python 3.10+
- **Key Libraries**: pandas, scikit-learn, plotly
- **Update Frequency**: Daily at 1:00 PM UTC (Mon-Fri)
- **Deployment**: GitHub Pages (static HTML)

## Files

```
fx_equity_sector_dashboard/
├── generate_fx_equity_dashboard.py   # Main analysis script
├── fx_equity_sector_fair_value.html  # Main dashboard
├── {currency}_analysis.html          # Individual currency pages
├── fx_equity_sector_data.json        # Summary data export
├── requirements.txt                  # Python dependencies
└── .github/workflows/
    └── update_dashboard.yml          # Automated update workflow
```

## Contributing

Issues and pull requests welcome! Potential enhancements:

- Add forward-looking signals (momentum, trend indicators)
- Incorporate macro data (rates, PMIs, commodities)
- Multi-timeframe analysis (daily, weekly, monthly)
- Ensemble models (Ridge, Random Forest, XGBoost)

## License

MIT License - feel free to use for research or trading ideas.

## Disclaimer

This dashboard is for informational and educational purposes only. It is **not investment advice**. FX trading involves substantial risk. Always do your own research and consult a licensed financial advisor before making trading decisions.

## Acknowledgments

- Equity data from [Stooq](https://stooq.com)
- FX data pipeline: [EMFX Risk Diffusion](https://github.com/DataVizHonduran/EMFX_risk_diffusion)
- Built with [Plotly](https://plotly.com/python/) and [scikit-learn](https://scikit-learn.org/)

---

**Author**: DataVizHonduran
**Last Updated**: 2025-12-06
