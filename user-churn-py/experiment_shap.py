"""
SHAP 기반 모델 해석 실험
- Random Forest(최고 성능 모델) 기준 SHAP 분석
- 출력물: summary plot, bar plot, 개별 유저 force plot 사례
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap

# 한글 폰트 설정 (DejaVu Sans fallback: 음수 기호 등 ASCII 문자 깨짐 방지)
plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import data_loader

warnings.filterwarnings("ignore")

# ── 설정 ─────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "matches"
OUTPUT_DIR = Path(__file__).parent / "results" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2015-05-25"
END_DATE = "2016-10-23"
ACTIVATION_PERIOD = 7
CHURN_OBSERVATION_PERIOD = 7
RANDOM_STATE = 42

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

# ── 데이터 로드 + 모델 학습 ──────────────────────────────────
print("데이터 로딩 중...")
raw_df = data_loader.load_parquet(str(DATA_DIR), START_DATE, END_DATE)
churn_col = f"ap_{ACTIVATION_PERIOD}d_and_cop_{CHURN_OBSERVATION_PERIOD}d"
result_df = data_loader.filter_df_with_names(raw_df, ACTIVATION_PERIOD, CHURN_OBSERVATION_PERIOD, churn_col)

names = result_df["name"]
X = result_df.drop(columns=["name", churn_col])
X.columns = [FEATURE_KO.get(c, c) for c in X.columns]
y = np.ravel(result_df[churn_col])

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, names, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"학습: {len(X_train):,}  |  테스트: {len(X_test):,}")

print("Random Forest 학습 중...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    class_weight="balanced", random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)
print(f"테스트 정확도: {rf.score(X_test, y_test):.4f}")

# ── SHAP 계산 ────────────────────────────────────────────────
print("SHAP 값 계산 중 (테스트 셋)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test)

# class 0(이탈) 기준 SHAP values
# shap_values.values shape: (n_samples, n_features, n_classes)
shap_churn = shap.Explanation(
    values=shap_values.values[:, :, 0],
    base_values=shap_values.base_values[:, 0],
    data=shap_values.data,
    feature_names=X_test.columns.tolist(),
)

# ── 1. Summary Plot (Beeswarm) ──────────────────────────────
print("Summary plot 생성 중...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_churn, show=False, plot_size=(10, 8))
plt.title("SHAP Summary Plot — 이탈(class=0) 기여도", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {OUTPUT_DIR / 'shap_summary.png'}")

# ── 2. Bar Plot (평균 |SHAP|) ────────────────────────────────
print("Bar plot 생성 중...")
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_churn, show=False)
plt.title("평균 |SHAP| — 피처별 이탈 예측 기여도", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {OUTPUT_DIR / 'shap_bar.png'}")

# ── 3. 개별 유저 사례 (이탈 확률 높은 유저 / 낮은 유저) ─────
proba_churn = rf.predict_proba(X_test)[:, 0]  # 이탈 확률

# 이탈 확률 가장 높은 유저
high_idx = np.argmax(proba_churn)
# 이탈 확률 가장 낮은 유저
low_idx = np.argmin(proba_churn)

for label, idx in [("high_risk", high_idx), ("low_risk", low_idx)]:
    user_name = names_test.iloc[idx]
    user_proba = proba_churn[idx]
    actual = "이탈" if y_test[idx] == 0 else "유지"

    print(f"\n{'=' * 60}")
    print(f"  {label}: {user_name} (이탈 확률: {user_proba:.2%}, 실제: {actual})")
    print(f"{'=' * 60}")

    # 피처값과 SHAP값 출력
    user_shap = shap_churn[idx]
    feat_df = pd.DataFrame({
        "피처": X_test.columns,
        "값": X_test.iloc[idx].values,
        "SHAP": user_shap.values,
    }).sort_values("SHAP", key=abs, ascending=False)
    print(feat_df.to_string(index=False))

    # Waterfall plot
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(user_shap, max_display=len(X_test.columns), show=False)
    plt.title(f"{user_name} — 이탈 확률 {user_proba:.2%} (실제: {actual})", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {OUTPUT_DIR / f'shap_waterfall_{label}.png'}")

print(f"\n모든 결과가 {OUTPUT_DIR}/ 에 저장되었습니다.")
