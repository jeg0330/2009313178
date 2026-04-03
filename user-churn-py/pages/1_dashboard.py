import datetime as _dt

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("Dashboard")

# ---------------------------------------------------------------------------
# Guard: ensure data has been loaded from the main app
# ---------------------------------------------------------------------------
if "cache_key" not in st.session_state:
    st.warning("Please configure settings on the main page first.")
    st.stop()

filtered_df = st.session_state.get("filtered_df")
raw_df = st.session_state.get("raw_df")
rf_model = st.session_state.get("rf_model")
feature_names = st.session_state.get("feature_names")
churn_col = st.session_state.get("churn_col")

# ---------------------------------------------------------------------------
# No data check
# ---------------------------------------------------------------------------
if raw_df is None or raw_df.empty or filtered_df is None or filtered_df.empty:
    st.warning("선택한 기간에 데이터가 없습니다")
    st.stop()

# ---------------------------------------------------------------------------
# 1. Churn / Retain summary metrics
# ---------------------------------------------------------------------------
st.header("Churn Summary")

churn_counts = filtered_df[churn_col].value_counts()
total = len(filtered_df)
churned = int(churn_counts.get(0, 0))
retained = int(churn_counts.get(1, 0))
churn_rate = churned / total * 100 if total > 0 else 0
retain_rate = retained / total * 100 if total > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Users", f"{total:,}")
col2.metric("Churned", f"{churned:,}", f"{churn_rate:.1f}%")
col3.metric("Retained", f"{retained:,}", f"{retain_rate:.1f}%")

# ---------------------------------------------------------------------------
# 2. Churn / Retain distribution bar chart
# ---------------------------------------------------------------------------
st.subheader("Churn Distribution")

dist_df = pd.DataFrame(
    {
        "Status": ["Churned", "Retained"],
        "Count": [churned, retained],
    }
)
fig_dist = px.bar(
    dist_df,
    x="Status",
    y="Count",
    color="Status",
    color_discrete_map={"Churned": "#EF553B", "Retained": "#636EFA"},
    text_auto=True,
)
fig_dist.update_layout(showlegend=False)
st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Feature importance bar chart (Random Forest)
# ---------------------------------------------------------------------------
if rf_model is not None and feature_names is not None:
    st.subheader("Feature Importance (Random Forest)")

    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".3f",
    )
    fig_fi.update_layout(yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig_fi, use_container_width=True)

# ---------------------------------------------------------------------------
# 4. Feature correlation heatmap
# ---------------------------------------------------------------------------
st.subheader("Feature Correlation Heatmap")

corr = filtered_df.corr(numeric_only=True)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
    )
)
fig_heatmap.update_layout(
    width=800,
    height=800,
    xaxis=dict(tickangle=-45),
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------------------------------------------------------------------------
# 5. Daily churn rate time-series line chart
# ---------------------------------------------------------------------------
st.subheader("Daily Churn Rate")

# Build per-date churn rate from the raw match data.
# A user's first_login_date determines their activation window; we attribute
# their churn label to that first_login_date for the time-series.
activation_period = st.session_state.get("activation_period", 7)
churn_observation_period = st.session_state.get("churn_observation_period", 7)

raw_copy = raw_df.copy()
raw_copy["first_login_date"] = raw_copy.groupby("name")["date"].transform("min")

ap_delta = _dt.timedelta(days=activation_period)
cop_delta = _dt.timedelta(days=churn_observation_period)

# Identify users who were active in the COP window
full = raw_copy[raw_copy["date"] < raw_copy["first_login_date"] + ap_delta + cop_delta].copy()
full["played_cop"] = (
    (full["date"] > full["first_login_date"] + ap_delta)
    & (full["date"] < full["first_login_date"] + ap_delta + cop_delta)
).astype(int)
user_labels = full.groupby("name").agg(
    first_login_date=("first_login_date", "first"),
    played_cop=("played_cop", "max"),
).reset_index()
user_labels["churned"] = (user_labels["played_cop"] == 0).astype(int)
user_labels["login_date"] = user_labels["first_login_date"].dt.date

daily = user_labels.groupby("login_date").agg(
    total=("churned", "count"),
    churned=("churned", "sum"),
).reset_index()
daily["churn_rate"] = daily["churned"] / daily["total"] * 100

fig_ts = px.line(
    daily,
    x="login_date",
    y="churn_rate",
    labels={"login_date": "Date", "churn_rate": "Churn Rate (%)"},
    markers=True,
)
fig_ts.update_layout(yaxis=dict(range=[0, 100]))
st.plotly_chart(fig_ts, use_container_width=True)
