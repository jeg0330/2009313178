"""
피처 선택 실험
- 다중공선성 분석 (VIF, 상관행렬)
- RFE (Recursive Feature Elimination) 로 피처 제거
- 전체 피처 vs 선택 피처 성능 비교
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
import warnings
import data_loader

warnings.filterwarnings("ignore")

# ── 설정 ─────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "matches"
START_DATE = "2015-05-25"
END_DATE = "2016-10-23"
ACTIVATION_PERIOD = 7
CHURN_OBSERVATION_PERIOD = 7
RANDOM_STATE = 42
K_FOLDS = 5

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
}

# ── 데이터 로드 ──────────────────────────────────────────────
print("데이터 로딩 중...")
raw_df = data_loader.load_parquet(str(DATA_DIR), START_DATE, END_DATE)
churn_col = f"ap_{ACTIVATION_PERIOD}d_and_cop_{CHURN_OBSERVATION_PERIOD}d"
result_df = data_loader.filter_df(raw_df, ACTIVATION_PERIOD, CHURN_OBSERVATION_PERIOD, churn_col)

X = result_df.drop(churn_col, axis=1)
y = np.ravel(result_df[churn_col])

print(f"유저 수: {len(y):,}  |  피처 수: {X.shape[1]}")

cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── 1. 상관행렬 분석 ─────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  1. 피처 간 상관관계 (|r| >= 0.7 인 쌍)")
print(f"{'=' * 80}")

corr = X.corr().abs()
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if corr.iloc[i, j] >= 0.7:
            high_corr_pairs.append({
                "피처 1": corr.columns[i],
                "피처 2": corr.columns[j],
                "|r|": corr.iloc[i, j],
            })

if high_corr_pairs:
    pair_df = pd.DataFrame(high_corr_pairs).sort_values("|r|", ascending=False)
    pair_df["피처 1 (한글)"] = pair_df["피처 1"].map(FEATURE_KO)
    pair_df["피처 2 (한글)"] = pair_df["피처 2"].map(FEATURE_KO)
    print(pair_df[["피처 1 (한글)", "피처 2 (한글)", "|r|"]].to_string(index=False))
else:
    print("  |r| >= 0.7 인 쌍 없음")

# ── 2. VIF 분석 ──────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  2. VIF (Variance Inflation Factor)")
print(f"{'=' * 80}")

from sklearn.linear_model import LinearRegression

vif_data = []
for i, col in enumerate(X.columns):
    others = X.drop(columns=[col])
    lr = LinearRegression()
    lr.fit(others, X[col])
    r2 = lr.score(others, X[col])
    vif = 1 / (1 - r2) if r2 < 1 else float("inf")
    vif_data.append({"피처": col, "한글": FEATURE_KO.get(col, col), "VIF": vif})

vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
print(vif_df[["한글", "VIF"]].to_string(index=False))
print(f"\n  VIF > 10: 다중공선성 높음, VIF > 5: 주의 필요")

high_vif = vif_df[vif_df["VIF"] > 10]["피처"].tolist()
if high_vif:
    print(f"  다중공선성 높은 피처: {[FEATURE_KO.get(f, f) for f in high_vif]}")

# ── 3. RFECV ─────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  3. RFECV (Recursive Feature Elimination with CV)")
print(f"{'=' * 80}")

rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    class_weight="balanced", random_state=RANDOM_STATE,
)

rfecv = RFECV(
    estimator=rf,
    step=1,
    cv=cv,
    scoring="roc_auc",
    min_features_to_select=3,
    n_jobs=-1,
)
rfecv.fit(X, y)

print(f"  최적 피처 수: {rfecv.n_features_}")
print(f"  선택된 피처:")
selected = X.columns[rfecv.support_].tolist()
eliminated = X.columns[~rfecv.support_].tolist()
for f in selected:
    print(f"    + {FEATURE_KO.get(f, f)} ({f})")
print(f"  제거된 피처:")
for f in eliminated:
    print(f"    - {FEATURE_KO.get(f, f)} ({f})")

# 피처 수별 CV 점수
print(f"\n  피처 수별 AUC-ROC:")
cv_results = rfecv.cv_results_
for i, score in enumerate(cv_results["mean_test_score"]):
    n_feat = i + 3  # min_features_to_select=3
    marker = " <<<" if n_feat == rfecv.n_features_ else ""
    print(f"    {n_feat:2d}개: {score:.4f} +/- {cv_results['std_test_score'][i]:.4f}{marker}")

# ── 4. 성능 비교 (전체 vs 선택) ──────────────────────────────
print(f"\n{'=' * 80}")
print(f"  4. 전체 피처 vs 선택 피처 성능 비교")
print(f"{'=' * 80}")

X_selected = X[selected]

configs = {
    f"전체 ({X.shape[1]}개)": X,
    f"RFECV 선택 ({len(selected)}개)": X_selected,
}

# VIF 높은 피처 제거한 세트도 비교
if high_vif:
    X_no_vif = X.drop(columns=high_vif)
    configs[f"VIF>10 제거 ({X_no_vif.shape[1]}개)"] = X_no_vif

print(f"\n{'구성':<25} {'Accuracy':>12} {'AUC-ROC':>12} {'F1 (macro)':>12}")
print(f"{'-' * 63}")

for name, X_subset in configs.items():
    acc = cross_val_score(rf, X_subset, y, cv=cv, scoring="accuracy").mean()
    auc = cross_val_score(rf, X_subset, y, cv=cv, scoring="roc_auc").mean()
    f1 = cross_val_score(rf, X_subset, y, cv=cv, scoring="f1_macro").mean()
    print(f"{name:<25} {acc:>12.4f} {auc:>12.4f} {f1:>12.4f}")
