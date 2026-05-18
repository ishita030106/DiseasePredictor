# Phase 1 — Dataset Creation & Structuring

## Step 1 — Finalize Diseases & Prediction Scope

### Objective
The forecasting model will analyze multiple disease-related indicators to predict possible outbreak risks for different regions.

### Selected Diseases
- Flu
- Dengue
- Malaria
- COVID-like symptoms

### Prediction Scope
The model will predict weekly outbreak risk levels for selected regions using historical and real-time multi-source data.

### Forecast Output
- Low Risk
- Medium Risk
- High Risk
- Predicted Outbreak Probability (%)

---

# Step 2 — Finalize Features

## Objective
The forecasting model will use multiple data signals and indicators to identify outbreak patterns and forecast future risks.

## Planned Features

### Search Trend Features
- Fever search trends
- Cough search trends
- Dengue symptom searches
- Flu symptom searches

### Weather Features
- Temperature
- Humidity
- Rainfall

### Health Features
- Weekly reported cases
- Weekly case growth rate

### Time & Location Features
- Week
- Region/Location

## Feature Purpose

| Feature Type | Purpose |
|---|---|
| Search Trends | Detect early public symptom patterns |
| Weather Data | Analyze environmental conditions affecting disease spread |
| Health Data | Track outbreak intensity and case growth |
| Time & Region | Identify temporal and regional outbreak patterns |

## Planned Time-Series Structure

The dataset will follow a weekly time-series structure for forecasting future outbreak risks.

### Example Structure

| Week | Region | Fever_Search | Humidity | Cases |
|---|---|---|---|---|
| 2025-W1 | Delhi | 78 | 82 | 120 |
| 2025-W2 | Delhi | 91 | 85 | 145 |

## Final Goal

Create a merged weekly time-series dataset that combines:
- Search trend data
- Weather/climate data
- Disease case data

This dataset will later be used for:
- Random Forest
- XGBoost
- LSTM
- Ensemble forecasting models
