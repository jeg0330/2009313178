import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from data_loader import load_parquet, filter_df_with_names, data_split
from model_training import (
    random_forest_classifier,
    knn_classifier,
    naive_bayes_classifier,
    xgboost_classifier,
    lightgbm_classifier,
)


@st.cache_data(show_spinner=False)
def _cached_load_parquet(data_dir, start_date, end_date):
    return load_parquet(data_dir, start_date, end_date)


@st.cache_data(show_spinner=False)
def _cached_filter(raw_df, activation_period, churn_observation_period, churn_col):
    return filter_df_with_names(
        raw_df,
        activation_period=activation_period,
        churn_observation_period=churn_observation_period,
        churn_column=churn_col,
    )

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TagPro 유저 이탈 대시보드", layout="wide")

components.html("""
<script>
(function() {
    const doc = window.parent.document;
    let overlay = doc.getElementById('loading-overlay');
    if (!overlay) {
        overlay = doc.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.style.cssText = 'display:none;position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:99999;justify-content:center;align-items:center;color:white;font-size:1.5rem;font-family:sans-serif;';
        overlay.textContent = '로딩 중...';
        doc.body.appendChild(overlay);
    }
    const observer = new MutationObserver(function() {
        const status = doc.querySelector('[data-testid="stStatusWidget"]');
        const visible = status && status.offsetHeight > 0 && status.querySelector('svg');
        overlay.style.display = visible ? 'flex' : 'none';
    });
    observer.observe(doc.body, { childList: true, subtree: true, attributes: true });
})();
</script>
""", height=0)

st.title("TagPro 유저 이탈 대시보드")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data" / "matches"

_parquet_files = sorted(DATA_DIR.glob("*.parquet")) if DATA_DIR.exists() else []
if _parquet_files:
    _min_date = datetime.date.fromisoformat(_parquet_files[0].stem)
    _max_date = datetime.date.fromisoformat(_parquet_files[-1].stem)
else:
    _min_date = datetime.date(2015, 5, 25)
    _max_date = datetime.date.today()

FEATURE_KO = {
    "score_mean": "평균 점수",
    "score_std": "점수 표준편차",
    "points_mean": "평균 포인트",
    "degree_mean": "평균 등급",
    "win_rate": "승률",
    "win_count": "승리 횟수",
    "lose_count": "패배 횟수",
    "winning_streak": "최대 연승",
    "losing_streak": "최대 연패",
    "game_count": "총 게임 수",
    "active_days": "활동 일수",
    "engagement_hours": "참여 시간(h)",
    "avg_gap_min": "평균 게임 간격(분)",
    "hour_std": "플레이 시간대 편차",
    "games_per_day": "일일 게임 수",
    "weekend_ratio": "주말 플레이 비율",
    "peak_hour_ratio": "피크타임 플레이 비율",
    "first_game_win": "첫 게임 승패",
    "comeback_after_loss": "3연패 후 복귀",
    "score_trend": "성적 추세(기울기)",
    "session_count": "세션 수",
    "games_per_session": "세션당 게임 수",
    "activity_decline": "활동 감소율",
}

FEATURE_COLS = list(FEATURE_KO.keys())

MODEL_NAMES = {
    "rf": "Random Forest",
    "knn": "KNN",
    "nb": "Naive Bayes",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

MODEL_FUNCS = {
    "rf": random_forest_classifier,
    "knn": knn_classifier,
    "nb": naive_bayes_classifier,
    "xgb": xgboost_classifier,
    "lgbm": lightgbm_classifier,
}

MODEL_INFO = {
    "rf": {
        "설명": "여러 개의 결정 트리를 앙상블하여 다수결로 분류하는 모델",
        "장점": "과적합에 강하고, 피처 중요도를 자체 제공하며, 하이퍼파라미터 튜닝 없이도 준수한 성능",
        "단점": "트리 수가 많으면 학습/예측 속도가 느려지고, 모델 해석이 단일 트리보다 어려움",
        "적합한 상황": "피처 수가 많고, 비선형 관계가 존재하는 데이터",
        "파라미터": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "class_weight": "balanced"},
    },
    "knn": {
        "설명": "새 데이터와 가장 가까운 k개의 이웃을 찾아 다수결로 분류하는 모델",
        "장점": "직관적이고 구현이 간단하며, 별도의 학습 과정이 필요 없음",
        "단점": "데이터 수가 많으면 예측이 느리고, 고차원 데이터에서 거리 계산이 무의미해질 수 있음",
        "적합한 상황": "데이터 수가 적고, 결정 경계가 불규칙한 경우",
        "파라미터": {"k": "CV로 자동 탐색 (1~15 홀수)", "cv": 7},
    },
    "nb": {
        "설명": "베이즈 정리 기반으로 각 클래스의 사후 확률을 계산하여 분류하는 모델",
        "장점": "매우 빠르고, 적은 데이터에서도 잘 동작하며, 확률 기반 해석 가능",
        "단점": "피처 간 독립 가정이 깨지면 성능 저하, 연속형 피처에 가우시안 분포 가정",
        "적합한 상황": "피처 간 독립성이 높고, 빠른 베이스라인이 필요한 경우",
        "파라미터": {"분포 가정": "Gaussian (연속형)"},
    },
    "xgb": {
        "설명": "그래디언트 부스팅 기반으로 약한 학습기를 순차적으로 보강하는 모델",
        "장점": "높은 예측 성능, 결측치 자체 처리, 규제(regularization) 내장",
        "단점": "하이퍼파라미터가 많아 튜닝이 복잡하고, 과적합 가능성",
        "적합한 상황": "정형 데이터에서 최고 성능이 필요한 경우",
        "파라미터": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "scale_pos_weight": "자동 계산"},
    },
    "lgbm": {
        "설명": "리프 중심 트리 분할 방식으로 빠르게 학습하는 그래디언트 부스팅 모델",
        "장점": "XGBoost보다 빠른 학습 속도, 대용량 데이터에 효율적, 메모리 사용량 적음",
        "단점": "적은 데이터에서 과적합되기 쉽고, 하이퍼파라미터 민감도 높음",
        "적합한 상황": "대규모 데이터에서 빠른 학습과 높은 성능이 동시에 필요한 경우",
        "파라미터": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "is_unbalance": True},
    },
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("설정")

date_range = st.sidebar.date_input(
    "날짜 범위",
    value=(_min_date, _max_date),
    min_value=_min_date,
    max_value=_max_date,
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range
    end_date = start_date

activation_period = st.sidebar.slider(
    "활성 기간 (일)", min_value=2, max_value=14, value=7
)

churn_observation_period = st.sidebar.slider(
    "이탈 관찰 기간 (일)", min_value=3, max_value=14, value=7
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
churn_col = f"ap_{activation_period}d_and_cop_{churn_observation_period}d"

raw_df = _cached_load_parquet(str(DATA_DIR), str(start_date), str(end_date))
filtered_df_names = (
    _cached_filter(raw_df, activation_period, churn_observation_period, churn_col)
    if not raw_df.empty else None
)
filtered_df = (
    filtered_df_names.drop(columns=["name"]) if filtered_df_names is not None and not filtered_df_names.empty
    else None
)

# 데이터가 바뀌면 모델 결과 초기화
_cache_key = (str(start_date), str(end_date), activation_period, churn_observation_period)
if st.session_state.get("cache_key") != _cache_key:
    st.session_state.trained_models = {}
    st.session_state.model_results = None
    st.session_state.X_train = None
    st.session_state.cache_key = _cache_key


def _ensure_split():
    """data split이 필요할 때만 실행한다."""
    if st.session_state.get("X_train") is not None:
        return
    X, y, X_train, X_test, y_train, y_test = data_split(
        filtered_df, churn_col, test_size=0.3, random_state=42, scale=False
    )
    st.session_state.feature_names = X.columns.tolist()
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test


def _ensure_model(key):
    """지정한 모델이 필요할 때만 학습한다."""
    cache = st.session_state.get("trained_models", {})
    if key in cache:
        return cache[key]
    _ensure_split()
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    _, _, model = MODEL_FUNCS[key](X_train, y_train, X_test, y_test)
    cache[key] = model
    st.session_state.trained_models = cache
    return model

# 피처 중요도를 지원하는 모델
IMPORTANCE_MODELS = {"rf", "xgb", "lgbm"}

if raw_df is None or raw_df.empty or filtered_df is None or filtered_df.empty:
    st.warning("선택한 기간에 데이터가 없습니다")
    st.stop()

# ---------------------------------------------------------------------------
# Header: 프로젝트 소개 + KPI
# ---------------------------------------------------------------------------
st.caption("TagPro 게임 유저의 이탈 패턴을 분석하고 머신러닝 모델로 이탈을 예측합니다.")

churn_counts = filtered_df[churn_col].value_counts()
total = len(filtered_df)
churned = int(churn_counts.get(0, 0))
retained = int(churn_counts.get(1, 0))
churn_rate = churned / total * 100 if total > 0 else 0
retain_rate = retained / total * 100 if total > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("분석 기간", f"{start_date} ~ {end_date}")
kpi2.metric("전체 유저", f"{total:,}")
kpi3.metric("이탈", f"{churned:,}", f"{churn_rate:.1f}%")
kpi4.metric("유지", f"{retained:,}", f"{retain_rate:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab0, tab1, tab2, tab3 = st.tabs(["데이터 설명", "대시보드", "이탈 예측", "모델 비교"])

# ============================= TAB 0: 데이터 설명 =============================
with tab0:
    st.header("원본 데이터 구조")
    st.markdown("TagPro 매치 로그에서 추출한 플레이어 단위 레코드입니다.")

    st.dataframe(
        pd.DataFrame([
            {"필드": "name", "타입": "string", "설명": "플레이어 이름 (인증된 유저만 포함)"},
            {"필드": "team", "타입": "string", "설명": "소속 팀 (Red / Blue)"},
            {"필드": "flair", "타입": "int", "설명": "플레어 보유 여부 (0: 없음, 1: 있음)"},
            {"필드": "score", "타입": "int", "설명": "매치에서 획득한 점수"},
            {"필드": "points", "타입": "int", "설명": "매치 종료 시 부여된 랭크 포인트"},
            {"필드": "degree", "타입": "int", "설명": "플레이어 등급 (경험치 기반)"},
            {"필드": "date", "타입": "datetime", "설명": "매치 시작 시각"},
            {"필드": "win", "타입": "float", "설명": "승리 여부 (1: 승, 0: 패, 0.5: 무승부)"},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("원본 데이터 미리보기")
    preview_n = st.slider("표시할 행 수", min_value=5, max_value=100, value=20, key="raw_preview_n")
    st.dataframe(raw_df.head(preview_n), use_container_width=True, hide_index=True)
    st.caption(f"전체 {len(raw_df):,}행 중 상위 {preview_n}행")

    st.divider()

    st.header("데이터 전처리")
    st.markdown("JSON 원본에서 Parquet으로 변환할 때 아래 전처리를 적용했습니다.")
    st.dataframe(
        pd.DataFrame([
            {"처리 항목": "비인증 유저 제거", "내용": "auth == False인 레코드 제외 (익명 유저는 추적 불가)"},
            {"처리 항목": "flair 이진화", "내용": "flair 값을 0(없음) / 1(있음)으로 변환"},
            {"처리 항목": "team 변환", "내용": "숫자 코드(1, 2)를 Red / Blue 문자열로 변환"},
            {"처리 항목": "win 계산", "내용": "팀별 점수를 비교하여 승리(1) / 패배(0) / 무승부(0.5) 산출"},
            {"처리 항목": "date 변환", "내용": "UNIX timestamp를 datetime으로 변환"},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.header("피처 엔지니어링")
    st.markdown(
        "원본 데이터를 유저 단위로 집계하여 아래 피처를 생성합니다. "
        "**활성 기간(AP)** 내 데이터만 사용하여 데이터 누수를 방지합니다."
    )

    st.dataframe(
        pd.DataFrame([
            {"피처": "평균 점수", "원본 필드": "score", "집계 방식": "mean", "설명": "AP 내 매치 점수의 평균"},
            {"피처": "점수 표준편차", "원본 필드": "score", "집계 방식": "std", "설명": "매치 점수의 변동 폭"},
            {"피처": "평균 포인트", "원본 필드": "points", "집계 방식": "mean", "설명": "랭크 포인트의 평균"},
            {"피처": "평균 등급", "원본 필드": "degree", "집계 방식": "mean", "설명": "플레이어 등급의 평균"},
            {"피처": "승률", "원본 필드": "win", "집계 방식": "mean", "설명": "전체 매치 중 승리 비율"},
            {"피처": "승리 횟수", "원본 필드": "win", "집계 방식": "sum", "설명": "총 승리 매치 수"},
            {"피처": "패배 횟수", "원본 필드": "win", "집계 방식": "sum(1-win)", "설명": "총 패배 매치 수"},
            {"피처": "최대 연승", "원본 필드": "win", "집계 방식": "streak max", "설명": "연속 승리 최대 횟수"},
            {"피처": "최대 연패", "원본 필드": "win", "집계 방식": "streak max", "설명": "연속 패배 최대 횟수"},
            {"피처": "총 게임 수", "원본 필드": "-", "집계 방식": "count", "설명": "AP 내 총 매치 참여 수"},
            {"피처": "활동 일수", "원본 필드": "date", "집계 방식": "nunique(date)", "설명": "플레이한 고유 날짜 수"},
            {"피처": "참여 시간(h)", "원본 필드": "date", "집계 방식": "max-min", "설명": "첫 게임~마지막 게임 경과 시간"},
            {"피처": "평균 게임 간격(분)", "원본 필드": "date", "집계 방식": "mean(gap)", "설명": "연속 매치 사이 평균 대기 시간"},
            {"피처": "플레이 시간대 편차", "원본 필드": "date", "집계 방식": "std(hour)", "설명": "플레이하는 시간대의 분산 (규칙적↓ 불규칙↑)"},
            {"피처": "일일 게임 수", "원본 필드": "-", "집계 방식": "count/days", "설명": "활동일 당 평균 매치 수"},
            {"피처": "주말 플레이 비율", "원본 필드": "date", "집계 방식": "mean(주말)", "설명": "토·일에 플레이한 비율"},
            {"피처": "피크타임 플레이 비율", "원본 필드": "date", "집계 방식": "mean(18~23시)", "설명": "피크타임(18~23시) 플레이 비율"},
            {"피처": "첫 게임 승패", "원본 필드": "win", "집계 방식": "first", "설명": "첫 게임 승리 여부 (1=승, 0=패/무)"},
            {"피처": "3연패 후 복귀", "원본 필드": "win", "집계 방식": "streak+flag", "설명": "3연패 이상 후에도 계속 플레이 여부"},
            {"피처": "성적 추세(기울기)", "원본 필드": "score", "집계 방식": "회귀 기울기", "설명": "게임 순번 대비 점수 변화 방향 (+:상승 -:하락)"},
            {"피처": "세션 수", "원본 필드": "date", "집계 방식": "nunique(session)", "설명": "30분 이상 간격으로 구분한 플레이 세션 수"},
            {"피처": "세션당 게임 수", "원본 필드": "-", "집계 방식": "count/sessions", "설명": "한 세션에서 평균적으로 플레이한 매치 수"},
            {"피처": "활동 감소율", "원본 필드": "date", "집계 방식": "(전반-후반)/전체", "설명": "AP 전반부 대비 후반부 게임 수 감소 비율 (+:감소 -:증가)"},
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.header("이탈 라벨 정의")
    st.markdown(f"""
- **활성 기간 (AP)**: 첫 접속일로부터 **{activation_period}일** — 이 기간의 데이터로 피처 생성
- **이탈 관찰 기간 (COP)**: AP 종료 후 **{churn_observation_period}일** — 이 기간에 접속 여부로 이탈 판정
- **이탈 (0)**: COP 기간에 한 번도 접속하지 않은 유저
- **유지 (1)**: COP 기간에 1회 이상 접속한 유저
""")

# ============================= TAB 1: 대시보드 =============================
with tab1:
    # --- 1. Churn Distribution ---
    st.subheader("이탈/유지 분포")

    dist_df = pd.DataFrame({"상태": ["이탈", "유지"], "인원": [churned, retained]})
    fig_dist = px.bar(
        dist_df, x="상태", y="인원", color="상태",
        color_discrete_map={"이탈": "#EF553B", "유지": "#636EFA"},
        text_auto=True,
    )
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- 3. Feature Importance ---
    importance_keys = sorted(IMPORTANCE_MODELS)
    dash_model_key = st.selectbox(
        "피처 중요도 모델",
        options=importance_keys,
        format_func=lambda k: MODEL_NAMES[k],
        key="dash_model",
    )

    model = _ensure_model(dash_model_key)
    feature_names = st.session_state.feature_names
    st.subheader(f"피처 중요도 ({MODEL_NAMES[dash_model_key]})")

    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {"피처": [FEATURE_KO.get(f, f) for f in feature_names], "중요도": importances}
    ).sort_values("중요도", ascending=True)

    fig_fi = px.bar(fi_df, x="중요도", y="피처", orientation="h", text_auto=".3f")
    fig_fi.update_layout(yaxis_title="", xaxis_title="중요도")
    st.plotly_chart(fig_fi, use_container_width=True)

    # --- 4. Feature Correlation Heatmap ---
    st.subheader("피처 상관관계 히트맵")

    corr = filtered_df.corr(numeric_only=True)
    corr.rename(columns=FEATURE_KO, index=FEATURE_KO, inplace=True)
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0,
            text=corr.round(2).values, texttemplate="%{text}", textfont={"size": 10},
        )
    )
    fig_heatmap.update_layout(width=800, height=800, xaxis=dict(tickangle=-45))
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- 5. Daily Churn Rate ---
    st.subheader("일별 이탈률")

    raw_copy = raw_df.copy()
    raw_copy["first_login_date"] = raw_copy.groupby("name")["date"].transform("min")

    ap_delta = datetime.timedelta(days=activation_period)
    cop_delta = datetime.timedelta(days=churn_observation_period)

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
        daily, x="login_date", y="churn_rate",
        labels={"login_date": "날짜", "churn_rate": "이탈률 (%)"},
        markers=True,
    )
    fig_ts.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_ts, use_container_width=True)


# ============================= TAB 2: 이탈 예측 =============================
with tab2:
    st.header("이탈 예측")

    if filtered_df_names is None or filtered_df_names.empty:
        st.warning("이탈 예측에 사용할 데이터가 없습니다")
    else:
        pred_model_key = st.selectbox(
            "예측 모델",
            options=list(MODEL_NAMES.keys()),
            format_func=lambda k: MODEL_NAMES[k],
            key="pred_model",
        )
        pred_model = _ensure_model(pred_model_key)
        _pred_features = st.session_state.feature_names  # 학습 시 컬럼 순서 그대로 사용
        X_all = filtered_df_names[_pred_features]
        proba = pred_model.predict_proba(X_all)[:, 1]
        # proba 는 churn_col==1 (유지) 확률이므로 이탈 확률 = 1 - proba
        # churn_col: 0=이탈, 1=유지 이므로 class 1 = 유지
        # 이탈 확률 = P(class=0) = 1 - P(class=1)
        churn_proba = 1 - proba

        pred_df = filtered_df_names[["name"] + _pred_features].copy()
        pred_df["이탈 확률"] = churn_proba

        def _risk_label(p):
            if p >= 0.7:
                return "높음"
            elif p >= 0.4:
                return "중간"
            else:
                return "낮음"

        pred_df["위험 등급"] = pred_df["이탈 확률"].apply(_risk_label)
        pred_df = pred_df.sort_values("이탈 확률", ascending=False).reset_index(drop=True)

        # --- Top N 위험 유저 테이블 ---
        st.subheader("이탈 위험 유저 Top N")

        top_n = st.slider("표시할 유저 수", min_value=5, max_value=min(100, len(pred_df)), value=20, key="top_n")

        risk_options = ["높음", "중간", "낮음"]
        selected_risks = st.multiselect("위험 등급 필터", options=risk_options, default=risk_options, key="risk_filter")

        display_df = pred_df[pred_df["위험 등급"].isin(selected_risks)].head(top_n)

        # 표시용 데이터프레임 (이름, 이탈 확률, 위험 등급)
        st.dataframe(
            display_df[["name", "이탈 확률", "위험 등급"]]
            .rename(columns={"name": "유저 이름"})
            .style.format({"이탈 확률": "{:.2%}"})
            .applymap(
                lambda v: (
                    "background-color: #ffcccc" if v == "높음"
                    else "background-color: #fff3cd" if v == "중간"
                    else "background-color: #d4edda"
                ),
                subset=["위험 등급"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        # --- 유저 검색 ---
        st.subheader("유저 검색")
        search_name = st.text_input("유저 이름을 입력하세요", key="user_search")

        if search_name:
            user_row = pred_df[pred_df["name"] == search_name]
            if user_row.empty:
                st.warning("해당 유저를 찾을 수 없습니다")
            else:
                user = user_row.iloc[0]
                st.markdown(f"**유저:** {user['name']}")

                ucol1, ucol2 = st.columns(2)
                ucol1.metric("이탈 확률", f"{user['이탈 확률']:.2%}")
                ucol2.metric("위험 등급", user["위험 등급"])

                # 피처 상세 테이블
                st.markdown("**피처 상세**")
                feat_data = {FEATURE_KO.get(f, f): [user[f]] for f in _pred_features if f in user.index}
                feat_display = pd.DataFrame(feat_data).T.rename(columns={0: "값"})
                st.dataframe(feat_display, use_container_width=True)

                # 레이더 차트 (각 피처를 min-max 정규화하여 0~1 스케일)
                st.markdown("**피처 레이더 차트**")
                radar_features = _pred_features
                radar_labels = [FEATURE_KO.get(f, f) for f in radar_features]

                # min-max 정규화 (전체 데이터 기준)
                user_vals = []
                for f in radar_features:
                    col_min = pred_df[f].min()
                    col_max = pred_df[f].max()
                    if col_max - col_min > 0:
                        user_vals.append((user[f] - col_min) / (col_max - col_min))
                    else:
                        user_vals.append(0.5)

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=user_vals + [user_vals[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill="toself",
                    name=user["name"],
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, use_container_width=True)


# ============================= TAB 3: 모델 비교 =============================
with tab3:
    st.header("모델 비교")

    _ensure_split()
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")

    if X_train is None or X_test is None:
        st.warning("모델 학습에 필요한 데이터가 없습니다")
    else:
        # --- 모델 선택 ---
        all_keys = list(MODEL_NAMES.keys())
        selected_keys = st.multiselect(
            "비교할 모델 선택",
            options=all_keys,
            default=all_keys,
            format_func=lambda k: MODEL_NAMES[k],
            key="model_select",
        )

        if len(selected_keys) == 0:
            st.info("비교할 모델을 1개 이상 선택하세요.")
        else:
            def _train_selected_models(X_tr, y_tr, X_te, y_te, keys):
                results = {}
                y_te_np = np.ravel(y_te)
                for key in keys:
                    func = MODEL_FUNCS[key]
                    _, y_pred, model = func(X_tr, y_tr, X_te, y_te)
                    y_pred = np.ravel(y_pred)
                    y_proba = model.predict_proba(X_te)[:, 1]
                    fpr, tpr, _ = roc_curve(y_te_np, y_proba)
                    cm = confusion_matrix(y_te_np, y_pred)
                    report = classification_report(y_te_np, y_pred, target_names=["이탈", "유지"], output_dict=True)
                    results[key] = {
                        "model": model,
                        "y_pred": y_pred,
                        "y_proba": y_proba,
                        "fpr": fpr,
                        "tpr": tpr,
                        "accuracy": accuracy_score(y_te_np, y_pred),
                        "auc_roc": roc_auc_score(y_te_np, y_proba),
                        "f1": f1_score(y_te_np, y_pred, average="macro"),
                        "precision": precision_score(y_te_np, y_pred, average="macro"),
                        "recall": recall_score(y_te_np, y_pred, average="macro"),
                        "confusion_matrix": cm,
                        "report": report,
                    }
                return results

            if st.button("모델 재학습", key="retrain_btn"):
                with st.spinner("모델 학습 중..."):
                    model_results = _train_selected_models(X_train, y_train, X_test, y_test, selected_keys)
                    st.session_state.model_results = model_results
                    st.session_state.trained_keys = selected_keys
                st.success("학습 완료!")

            model_results = st.session_state.get("model_results")
            trained_keys = st.session_state.get("trained_keys", [])
            available_keys = [k for k in selected_keys if k in trained_keys and model_results and k in model_results]

            if not available_keys:
                st.info("'모델 재학습' 버튼을 눌러 선택한 모델을 학습하세요.")
            else:
                # --- 성능 비교 표 ---
                st.subheader("성능 비교")

                perf_rows = []
                for key in available_keys:
                    r = model_results[key]
                    perf_rows.append({
                        "모델": MODEL_NAMES[key],
                        "정확도": r["accuracy"],
                        "AUC-ROC": r["auc_roc"],
                        "F1": r["f1"],
                        "Precision": r["precision"],
                        "Recall": r["recall"],
                    })

                perf_df = pd.DataFrame(perf_rows)
                metric_cols = ["정확도", "AUC-ROC", "F1", "Precision", "Recall"]
                st.dataframe(
                    perf_df.style.format({c: "{:.4f}" for c in metric_cols})
                    .highlight_max(subset=metric_cols, color="#d4edda"),
                    use_container_width=True,
                    hide_index=True,
                )

                # --- ROC Curve 비교 ---
                if len(available_keys) >= 2:
                    st.subheader("ROC Curve 비교")

                    fig_roc = go.Figure()
                    for key in available_keys:
                        r = model_results[key]
                        fig_roc.add_trace(go.Scatter(
                            x=r["fpr"], y=r["tpr"], mode="lines",
                            name=f"{MODEL_NAMES[key]} (AUC={r['auc_roc']:.4f})",
                        ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="랜덤 기준선", line=dict(dash="dash", color="gray"),
                    ))
                    fig_roc.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        legend=dict(x=0.5, y=0.02, xanchor="center"),
                        height=500,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                # --- 모델별 상세 ---
                st.subheader("모델별 상세")

                for key in available_keys:
                    r = model_results[key]
                    info = MODEL_INFO[key]

                    with st.expander(f"{MODEL_NAMES[key]}", expanded=len(available_keys) == 1):
                        # 설명
                        st.markdown(f"**알고리즘:** {info['설명']}")
                        dcol1, dcol2 = st.columns(2)
                        dcol1.markdown(f"**장점:** {info['장점']}")
                        dcol2.markdown(f"**단점:** {info['단점']}")
                        st.markdown(f"**적합한 상황:** {info['적합한 상황']}")

                        # 파라미터
                        st.markdown("**하이퍼파라미터**")
                        param_df = pd.DataFrame(
                            [{"파라미터": k, "값": str(v)} for k, v in info["파라미터"].items()]
                        )
                        st.dataframe(param_df, use_container_width=True, hide_index=True)

                        st.divider()

                        # 혼동 행렬 + 분류 리포트 나란히
                        cm_col, rpt_col = st.columns(2)

                        with cm_col:
                            st.markdown("**혼동 행렬**")
                            cm = r["confusion_matrix"]
                            fig_cm = go.Figure(data=go.Heatmap(
                                z=cm, x=["이탈 (예측)", "유지 (예측)"], y=["이탈 (실제)", "유지 (실제)"],
                                colorscale="Blues",
                                text=cm, texttemplate="%{text}",
                                textfont={"size": 16},
                                showscale=False,
                            ))
                            fig_cm.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig_cm, use_container_width=True)

                        with rpt_col:
                            st.markdown("**분류 리포트**")
                            report = r["report"]
                            rpt_rows = []
                            for label in ["이탈", "유지"]:
                                rpt_rows.append({
                                    "클래스": label,
                                    "Precision": report[label]["precision"],
                                    "Recall": report[label]["recall"],
                                    "F1-Score": report[label]["f1-score"],
                                    "Support": int(report[label]["support"]),
                                })
                            rpt_df = pd.DataFrame(rpt_rows)
                            st.dataframe(
                                rpt_df.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
                                use_container_width=True,
                                hide_index=True,
                            )
