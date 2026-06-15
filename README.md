# 🏬 Mall Footfall Intelligence

> **ML-powered dashboard** to forecast and analyze mall visitor traffic using hourly patterns, demographics, and time-series features.

---

## 📸 Features

| Page | Description |
|------|-------------|
| 📊 EDA Dashboard | Interactive charts: visits per mall, hourly patterns, gender/age/ARPU breakdown |
| 🔮 Forecasting | Predict hourly footfall for any mall up to 72 hours ahead |
| 📈 Model Performance | R², MAE, RMSE, actual vs predicted, feature importance, per-mall error |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/mall-footfall-intelligence.git
cd mall-footfall-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your data
Put `total_visitors_per_mall_cleaned.csv` in the root folder (not committed — see `.gitignore`).

### 4. Train the model
```bash
python footfall_full_pipeline.py
```
This generates:
- `footfall_model.pkl` — trained Random Forest model
- `feature_cols.pkl` — feature names in correct order
- `hourly_features.csv` — aggregated hourly data used by the app

### 5. Run the app
```bash
streamlit run app.py
```
Opens at → `http://localhost:8501`

---

## 📁 Project Structure

```
mall-footfall-intelligence/
│
├── app.py                        # Streamlit production app (3 pages)
├── footfall_full_pipeline.py     # Data prep + feature engineering + model training
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes data & model files
└── README.md                     # You are here
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Trees | 300 |
| Max Depth | 12 |
| Target | Hourly visitor count per mall |
| R² Score | **0.95** |
| MAE | **~29 visitors/hour** |
| RMSE | **~65 visitors/hour** |

### Features used
- **Temporal:** hour, day of week, day of month, is_weekend
- **Cyclical encoding:** hour_sin/cos, dow_sin/cos
- **Lag features:** lag_1h, lag_24h, lag_168h (1 week)
- **Rolling average:** 24-hour rolling mean
- **Mall identity:** one-hot encoded mall name

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push the code to GitHub (data & model files are gitignored)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as the entry point
4. Add your CSV as a **Secret file** or use cloud storage

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `scikit-learn` · `Plotly` · `Pandas` · `NumPy` · `Joblib`

---

## 👤 Author

Emad Alqudah

Built as a portfolio project demonstrating end-to-end ML: EDA → Feature Engineering → Model Training → Production Deployment.