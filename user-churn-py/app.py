import datetime
from pathlib import Path

import streamlit as st

from data_loader import load_parquet, filter_df, data_split
from model_training import random_forest_classifier

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TagPro User Churn Dashboard", layout="wide")
st.title("TagPro User Churn Dashboard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data" / "matches"

# Discover available date range from parquet files
_parquet_files = sorted(DATA_DIR.glob("*.parquet")) if DATA_DIR.exists() else []
if _parquet_files:
    _min_date = datetime.date.fromisoformat(_parquet_files[0].stem)
    _max_date = datetime.date.fromisoformat(_parquet_files[-1].stem)
else:
    _min_date = datetime.date(2015, 5, 25)
    _max_date = datetime.date.today()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

date_range = st.sidebar.date_input(
    "Date Range",
    value=(_min_date, _max_date),
    min_value=_min_date,
    max_value=_max_date,
)

# date_input can return a single date when the user hasn't finished selecting
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range
    end_date = start_date

activation_period = st.sidebar.slider(
    "Activation Period (days)", min_value=2, max_value=14, value=7
)

churn_observation_period = st.sidebar.slider(
    "Churn Observation Period (days)", min_value=3, max_value=14, value=7
)

# ---------------------------------------------------------------------------
# Data / model caching via session_state
# ---------------------------------------------------------------------------
_cache_key = (str(start_date), str(end_date), activation_period, churn_observation_period)

if "cache_key" not in st.session_state or st.session_state.cache_key != _cache_key:
    raw_df = load_parquet(str(DATA_DIR), str(start_date), str(end_date))

    if raw_df.empty:
        st.session_state.raw_df = raw_df
        st.session_state.filtered_df = None
        st.session_state.rf_model = None
        st.session_state.feature_names = None
        st.session_state.X = None
        st.session_state.y = None
    else:
        churn_col = f"ap_{activation_period}d_and_cop_{churn_observation_period}d"
        filtered_df = filter_df(
            raw_df,
            activation_period=activation_period,
            churn_observation_period=churn_observation_period,
            churn_column=churn_col,
        )

        if filtered_df.empty or churn_col not in filtered_df.columns:
            st.session_state.raw_df = raw_df
            st.session_state.filtered_df = None
            st.session_state.rf_model = None
            st.session_state.feature_names = None
            st.session_state.X = None
            st.session_state.y = None
        else:
            X, y, X_train, X_test, y_train, y_test = data_split(
                filtered_df, churn_col, test_size=0.3, random_state=42, scale=False
            )
            _, _, rf_model = random_forest_classifier(X_train, y_train, X_test, y_test)

            st.session_state.raw_df = raw_df
            st.session_state.filtered_df = filtered_df
            st.session_state.rf_model = rf_model
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.churn_col = churn_col

    st.session_state.cache_key = _cache_key
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    st.session_state.activation_period = activation_period
    st.session_state.churn_observation_period = churn_observation_period

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------
st.markdown(
    """
    Use the **sidebar** to configure the date range and churn parameters,
    then navigate to the **Dashboard** page to view analytics.
    """
)
