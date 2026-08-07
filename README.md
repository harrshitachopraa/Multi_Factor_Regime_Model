# 📈 Macro Regime-Based Factor Rotation Prediction

Predicting the next month's **relative performance of Momentum, Quality, and Value factors** using macroeconomic indicators and machine learning.

> Built for a quantitative finance competition focused on tactical factor allocation and macro regime forecasting.

---

## 📌 Project Overview

Financial markets constantly rotate leadership between investment styles such as **Momentum**, **Quality**, and **Value** depending on the prevailing macroeconomic environment.

This project develops a **regime-aware machine learning model** that forecasts the **1-month forward ranking** of three Nifty factor indices using only macroeconomic and market-structure indicators.

Instead of predicting returns directly, the model learns to identify **which factor is most likely to outperform under the current macro regime**.

---

## 🎯 Objective

For every month-end observation, predict the ranking of:

| Factor | NSE Index |
|---------|-----------|
| Momentum | Nifty200 Momentum 30 |
| Quality | NIFTY200 Quality 30 |
| Value | Nifty200 Value 30 |

Output format:

| Rank | Meaning |
|------|---------|
| 1 | Highest expected return |
| 2 | Middle |
| 3 | Lowest expected return |

Each prediction must be a valid permutation of **{1,2,3}**.

---

## 📊 Dataset

The dataset contains monthly observations from **October 2015 onwards**.

### Features

Approximately **40 macroeconomic indicators**, including:

- Global Liquidity
- Federal Reserve Balance Sheet
- US Treasury Yields
- Credit Spreads
- US CPI
- Payroll Data
- India FII/DII Flows
- Commodities (Gold, Silver, Copper, Crude Oil)
- Dollar Index
- Market Volatility
- Market Momentum
- Valuation Indicators
- Earnings Yield Gap
- Buffett Indicator
- Risk Sentiment Measures

Target variables:

- Forward rank of Momentum
- Forward rank of Quality
- Forward rank of Value

---

## 🧠 Methodology

This solution uses **XGBoost Listwise Learning-to-Rank** instead of traditional regression or classification.

### Feature Engineering

Several causal (time-safe) features were created:

- 3-Month Momentum
- 3-Month Moving Average
- 6-Month Rolling Volatility
- Interaction Features
- Commodity Strength Index
- Risk Appetite Indicator
- Expanding Regime Indicators
- Credit Stress Signals
- Fed Tightening Regime
- High Yield Regime
- High Volatility Regime

All engineered features are strictly **causal**, ensuring no future information leaks into training.

---

## 🔄 Learning-to-Rank Framework

Each month is transformed into **three observations**:

- Momentum
- Quality
- Value

Each factor receives:

- Same macro features
- Factor identity flag

The model then learns to rank the three factors directly using:

```
objective = rank:pairwise
```

instead of predicting returns independently.

---

## ⚙️ Model

**Algorithm**

- XGBoost Ranker

Key Parameters

```python
objective = "rank:pairwise"
max_depth = 3
learning_rate = 0.05
n_estimators = 80
min_child_weight = 4
```

---

## 📈 Validation Strategy

A **Walk-Forward Cross Validation** framework was used to respect the chronological structure of financial time series.

Evaluation Metric:

**Mean Monthly Spearman Rank Correlation**

This measures how closely the predicted factor ordering matches the true ordering.

---

## 🏆 Performance

Walk-forward validation achieved a mean **Spearman Rank Correlation of approximately 0.213**, demonstrating that the model captures meaningful relationships between macroeconomic regimes and future factor leadership. :contentReference[oaicite:0]{index=0}

---

## 📂 Project Structure

```
├── train.csv
├── test.csv
├── submit_1_xgb_ranker.py
└── README.md
```

---

## 🚀 Running the Project

Install dependencies

```bash
pip install pandas numpy xgboost scikit-learn
```

Run

```bash
python submit_1_xgb_ranker.py
```

Output

```
submission_1_xgb_ranker.csv
```

---

## 📚 Skills Demonstrated

- Quantitative Finance
- Factor Investing
- Tactical Asset Allocation
- Machine Learning
- Learning to Rank (LTR)
- Time-Series Feature Engineering
- Financial Feature Engineering
- XGBoost
- Cross-Sectional Ranking
- Walk-Forward Validation
- Python
- Pandas
- NumPy

---

## 🔮 Future Improvements

- LightGBM Ranker
- CatBoost Ranking
- Transformer-based Time Series Models
- Hidden Markov Models for Regime Detection
- Ensemble Ranking Models
- Bayesian Optimization
- SHAP-based Feature Interpretation

---

## 💡 Key Takeaway

This project demonstrates how macroeconomic indicators can be transformed into actionable investment signals using **Learning-to-Rank** techniques. Rather than forecasting absolute returns, the model predicts the **relative leadership of investment styles**, making it directly applicable to tactical portfolio allocation and systematic factor investing.

---

## 👩‍💻 Author

**Harshita Chopra**

B.Tech, IIT (BHU) Varanasi

Interested in:
- Quantitative Research
- Machine Learning
- Portfolio Optimization
- Factor Investing
- Financial AI
