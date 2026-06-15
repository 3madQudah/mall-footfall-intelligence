"""
app.py — Mall Footfall Forecasting | Production
=================================================
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Footfall Intelligence",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid #6366f1;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #6366f1; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #6366f1; margin: 1.5rem 0 0.5rem;
        border-bottom: 1px solid #334155; padding-bottom: 6px;
    }
    div[data-testid="stMetric"] {
        background: #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 3px solid #6366f1;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #34d399 !important;
    }
</style>
""", unsafe_allow_html=True)

ACCENT   = "#6366f1"
BG_DARK  = "#0f172a"
BG_CARD  = "#1e293b"
TEXT_MUT = "#94a3b8"
PLOTLY_TEMPLATE = "plotly_dark"

# ── Data & model loaders ───────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw():
    df = pd.read_csv("/Users/mac/Desktop/Zain Projects/Task 2 Visitor Mall Dataset/data/processed/total_visitors_per_mall_cleaned.csv")
    df["VISIT_DATE"] = pd.to_datetime(df["VISIT_DATE"])
    df["HOUR"]    = df["VISIT_DATE"].dt.hour
    df["WEEKDAY"] = df["VISIT_DATE"].dt.day_name()
    return df

@st.cache_data(show_spinner=False)
def load_hourly():
    return pd.read_csv("/Users/mac/Desktop/Zain Projects/Task 2 Visitor Mall Dataset/data/hourly_features/hourly_features.csv", parse_dates=["HOUR_TS"])

@st.cache_resource(show_spinner=False)
def load_model():
    model        = joblib.load("/Users/mac/Desktop/Zain Projects/Task 2 Visitor Mall Dataset/models/footfall_model.pkl")
    feature_cols = joblib.load("/Users/mac/Desktop/Zain Projects/Task 2 Visitor Mall Dataset/models/feature_cols.pkl")
    return model, feature_cols

raw     = load_raw()
hourly  = load_hourly()
model, feature_cols = load_model()

mall_cols  = [c for c in feature_cols if c.startswith("mall_")]
mall_names = [c.replace("mall_", "") for c in mall_cols]

WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏬 Mall Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 EDA Dashboard", "🔮 Forecasting", "📈 Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"<span style='color:{TEXT_MUT};font-size:0.8rem'>Data: Nov 2022 · 5 Malls<br>Model: Random Forest · R²=0.95</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 1 — EDA Dashboard
# ══════════════════════════════════════════════════════════
if page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    st.markdown(f"<span style='color:{TEXT_MUT}'>November 2022 · 899,536 visits · 5 malls</span>", unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Visits",   f"{len(raw):,}")
    k2.metric("Unique Visitors", f"{raw['SUBSCRIBER_ID'].nunique():,}")
    k3.metric("Malls",          f"{raw['MALL_NAME'].nunique()}")
    k4.metric("Peak Hour",      "8 PM")

    st.markdown("---")

    # Row 1 — visits per mall + weekday
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-header">Total Visits per Mall</p>', unsafe_allow_html=True)
        vc = raw["MALL_NAME"].value_counts().reset_index()
        vc.columns = ["Mall", "Visits"]
        fig = px.bar(vc, x="Visits", y="Mall", orientation="h",
                     color="Visits", color_continuous_scale="Viridis",
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Visits by Day of Week</p>', unsafe_allow_html=True)
        wd = raw["WEEKDAY"].value_counts().reindex(WEEKDAY_ORDER).reset_index()
        wd.columns = ["Day", "Visits"]
        fig = px.bar(wd, x="Day", y="Visits", color="Visits",
                     color_continuous_scale="Plasma", template=PLOTLY_TEMPLATE)
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — hourly pattern + age
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-header">Visits by Hour of Day</p>', unsafe_allow_html=True)
        hv = raw["HOUR"].value_counts().sort_index().reset_index()
        hv.columns = ["Hour", "Visits"]
        fig = px.area(hv, x="Hour", y="Visits", template=PLOTLY_TEMPLATE,
                      color_discrete_sequence=[ACCENT])
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<p class="section-header">Age Bracket Distribution</p>', unsafe_allow_html=True)
        ag = raw["AGE_BRACKET"].value_counts().reset_index()
        ag.columns = ["Age", "Visits"]
        fig = px.pie(ag, names="Age", values="Visits", hole=0.45,
                     template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3 — gender per mall + ARPU heatmap
    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<p class="section-header">Gender Distribution per Mall</p>', unsafe_allow_html=True)
        gm = raw.groupby(["MALL_NAME","GENDER"]).size().reset_index(name="Visits")
        fig = px.bar(gm, x="MALL_NAME", y="Visits", color="GENDER",
                     barmode="group", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=[ACCENT, "#f472b6"])
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280,
                          xaxis_title="", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown('<p class="section-header">ARPU Bracket by Mall (%)</p>', unsafe_allow_html=True)
        ct = pd.crosstab(raw["MALL_NAME"], raw["ARPU_BRACKET"], normalize="index").round(3) * 100
        fig = px.imshow(ct, text_auto=".1f", color_continuous_scale="Blues",
                        template=PLOTLY_TEMPLATE, aspect="auto")
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

    # Row 4 — Device OS
    st.markdown('<p class="section-header">Device OS Split</p>', unsafe_allow_html=True)
    os_vc = raw["DEVICE_OS"].value_counts().reset_index()
    os_vc.columns = ["OS", "Visits"]
    fig = px.bar(os_vc, x="OS", y="Visits", color="OS",
                 template=PLOTLY_TEMPLATE,
                 color_discrete_sequence=[ACCENT,"#f472b6","#34d399"])
    fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0), height=250)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — Forecasting
# ══════════════════════════════════════════════════════════
elif page == "🔮 Forecasting":
    st.title("🔮 Footfall Forecasting")
    st.markdown(f"<span style='color:{TEXT_MUT}'>Predict hourly visitor counts using Random Forest (R²=0.95)</span>", unsafe_allow_html=True)

    # Controls
    ctrl1, ctrl2 = st.columns([1, 2])
    with ctrl1:
        selected_mall = st.selectbox("Select Mall", mall_names)
        horizon       = st.slider("Forecast Horizon (hours)", 6, 72, 24)

    # Build forecast
    def build_forecast(mall_name, horizon):
        col = f"mall_{mall_name}"
        mall_hist = hourly[hourly[col] == True].sort_values("HOUR_TS")
        last_time = mall_hist["HOUR_TS"].iloc[-1]
        visit_buf = list(mall_hist["visits"].tail(168).values)

        rows = []
        for step in range(1, horizon + 1):
            ts  = last_time + pd.Timedelta(hours=step)
            h   = ts.hour
            dow = ts.dayofweek
            lag1   = visit_buf[-1]
            lag24  = visit_buf[-24]  if len(visit_buf) >= 24  else lag1
            lag168 = visit_buf[-168] if len(visit_buf) >= 168 else lag1
            roll   = float(np.mean(visit_buf[-24:]))

            row = {
                "hour": h, "dayofweek": dow, "day": ts.day,
                "is_weekend": int(dow in [4,5]),
                "hour_sin": np.sin(2*np.pi*h/24), "hour_cos": np.cos(2*np.pi*h/24),
                "dow_sin":  np.sin(2*np.pi*dow/7),"dow_cos":  np.cos(2*np.pi*dow/7),
                "lag_1": lag1, "lag_24": lag24, "lag_168": lag168, "rolling_24h": roll,
            }
            for mc in mall_cols:
                row[mc] = (mc == col)

            pred = float(np.clip(model.predict(pd.DataFrame([row])[feature_cols])[0], 0, None))
            row["predicted_visits"] = pred
            row["HOUR_TS"] = ts
            rows.append(row)
            visit_buf.append(pred)

        return pd.DataFrame(rows)

    with st.spinner("Generating forecast..."):
        fdf = build_forecast(selected_mall, horizon)

    # KPI row
    peak = fdf.loc[fdf["predicted_visits"].idxmax()]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Next Hour Forecast", f"{fdf.iloc[0]['predicted_visits']:.0f}")
    m2.metric("Avg / Hour",         f"{fdf['predicted_visits'].mean():.0f}")
    m3.metric("Peak Visitors",      f"{fdf['predicted_visits'].max():.0f}")
    m4.metric("Peak Time",          peak["HOUR_TS"].strftime("%a %H:%M"))

    # Forecast chart
    st.markdown('<p class="section-header">Forecast Timeline</p>', unsafe_allow_html=True)

    # Actual last 72h + forecast
    col = f"mall_{selected_mall}"
    actual = hourly[hourly[col] == True].sort_values("HOUR_TS").tail(72)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual["HOUR_TS"], y=actual["visits"],
        name="Actual", line=dict(color="#94a3b8", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fdf["HOUR_TS"], y=fdf["predicted_visits"],
        name="Forecast", line=dict(color=ACCENT, width=2.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0,r=0,t=30,b=0), height=380,
        xaxis_title="Time", yaxis_title="Visitors"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hourly breakdown table
    st.markdown('<p class="section-header">Hourly Breakdown</p>', unsafe_allow_html=True)
    table = fdf[["HOUR_TS","predicted_visits"]].copy()
    table.columns = ["Time", "Predicted Visitors"]
    table["Time"] = table["Time"].dt.strftime("%a %d %b · %H:%M")
    table["Predicted Visitors"] = table["Predicted Visitors"].round(0).astype(int)
    st.dataframe(table, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown(f"<span style='color:{TEXT_MUT}'>Random Forest · 300 trees · Train/Test split (last 7 days = test)</span>", unsafe_allow_html=True)

    # Compute metrics on test set
    hourly_s = hourly.sort_values("HOUR_TS").reset_index(drop=True)
    X = hourly_s[feature_cols]
    y = hourly_s["visits"]
    split = len(hourly_s) - 168 * len(mall_cols)
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    preds = np.clip(model.predict(X_test), 0, None)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",  f"{r2:.4f}",  "94.85% variance explained")
    m2.metric("MAE",       f"{mae:.1f}", "avg error per hour")
    m3.metric("RMSE",      f"{rmse:.1f}","penalises peak errors")

    st.markdown("---")
    c1, c2 = st.columns(2)

    # Actual vs Predicted scatter
    with c1:
        st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
        fig = px.scatter(
            x=y_test.values, y=preds,
            labels={"x":"Actual","y":"Predicted"},
            opacity=0.4, template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[ACCENT]
        )
        mn, mx = float(y_test.min()), float(y_test.max())
        fig.add_shape(type="line", x0=mn,y0=mn,x1=mx,y1=mx,
                      line=dict(color="#f472b6", dash="dash"))
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    with c2:
        st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
        imp = pd.Series(model.feature_importances_, index=feature_cols) \
                .sort_values(ascending=True).tail(10)
        fig = px.bar(
            x=imp.values, y=imp.index, orientation="h",
            color=imp.values, color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE,
            labels={"x":"Importance","y":"Feature"}
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Residuals over time
    st.markdown('<p class="section-header">Residuals over Test Period</p>', unsafe_allow_html=True)
    test_times = hourly_s["HOUR_TS"].iloc[split:].values
    residuals  = y_test.values - preds
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_times, y=residuals,
        mode="lines", line=dict(color=ACCENT, width=1),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.1)"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#f472b6")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Time", yaxis_title="Residual (Actual − Predicted)",
        margin=dict(l=0,r=0,t=10,b=0), height=280
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-mall error
    st.markdown('<p class="section-header">MAE per Mall (Test Set)</p>', unsafe_allow_html=True)
    test_df = hourly_s.iloc[split:].copy()
    test_df["pred"] = preds
    mall_errors = []
    for mc in mall_cols:
        mn = mc.replace("mall_","")
        sub = test_df[test_df[mc] == True]
        if len(sub):
            mall_errors.append({
                "Mall": mn,
                "MAE":  mean_absolute_error(sub["visits"], sub["pred"])
            })
    me_df = pd.DataFrame(mall_errors).sort_values("MAE")
    fig = px.bar(me_df, x="Mall", y="MAE", color="MAE",
                 color_continuous_scale="RdYlGn_r",
                 template=PLOTLY_TEMPLATE)
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      margin=dict(l=0,r=0,t=10,b=0), height=260)
    st.plotly_chart(fig, use_container_width=True)