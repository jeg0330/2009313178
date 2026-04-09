"""
하이퍼파라미터 튜닝 실험
- RandomizedSearchCV로 RF, XGBoost, LightGBM 탐색
- 탐색 범위 + 최적 파라미터 + 성능 비교 출력
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
N_ITER = 30  # RandomizedSearch 탐색 횟수

# ── 데이터 로드 ──────────────────────────────────────────────
print("데이터 로딩 중...")
raw_df = data_loader.load_parquet(str(DATA_DIR), START_DATE, END_DATE)
churn_col = f"ap_{ACTIVATION_PERIOD}d_and_cop_{CHURN_OBSERVATION_PERIOD}d"
result_df = data_loader.filter_df(raw_df, ACTIVATION_PERIOD, CHURN_OBSERVATION_PERIOD, churn_col)

X = result_df.drop(churn_col, axis=1)
y = np.ravel(result_df[churn_col])

print(f"유저 수: {len(y):,}  |  이탈 비율: {(y == 0).mean():.2%}")

cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── 탐색 범위 정의 ───────────────────────────────────────────
search_spaces = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample", None],
        },
    },
    "XGBoost": {
        "model": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
        },
    },
    "LightGBM": {
        "model": LGBMClassifier(
            is_unbalance=True, random_state=RANDOM_STATE, verbose=-1,
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 6, 8, 10, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63, 127],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [5, 10, 20],
        },
    },
}

# ── 탐색 범위 출력 ───────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  탐색 범위 (RandomizedSearchCV, n_iter={N_ITER}, {K_FOLDS}-fold CV)")
print(f"{'=' * 80}")
for name, cfg in search_spaces.items():
    print(f"\n  [{name}]")
    for param, values in cfg["params"].items():
        print(f"    {param}: {values}")

# ── 튜닝 실행 ────────────────────────────────────────────────
results = {}
for name, cfg in search_spaces.items():
    print(f"\n{'=' * 80}")
    print(f"  {name} 튜닝 중... (n_iter={N_ITER})")
    print(f"{'=' * 80}")

    search = RandomizedSearchCV(
        cfg["model"],
        cfg["params"],
        n_iter=N_ITER,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    search.fit(X, y)

    results[name] = search

    print(f"  최적 AUC-ROC: {search.best_score_:.4f}")
    print(f"  최적 파라미터:")
    for param, val in search.best_params_.items():
        print(f"    {param}: {val}")

# ── 기존 vs 튜닝 비교 ───────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  기존 파라미터 vs 튜닝 후 비교 (AUC-ROC, {K_FOLDS}-fold CV)")
print(f"{'=' * 80}")

from sklearn.model_selection import cross_val_score

baseline_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        eval_metric="logloss", random_state=RANDOM_STATE,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        is_unbalance=True, random_state=RANDOM_STATE, verbose=-1,
    ),
}

print(f"\n{'모델':<16} {'기존 AUC':>12} {'튜닝 AUC':>12} {'개선폭':>10}")
print(f"{'-' * 52}")

for name in search_spaces:
    baseline_auc = cross_val_score(
        baseline_models[name], X, y, cv=cv, scoring="roc_auc"
    ).mean()
    tuned_auc = results[name].best_score_
    delta = tuned_auc - baseline_auc
    print(f"{name:<16} {baseline_auc:>12.4f} {tuned_auc:>12.4f} {delta:>+10.4f}")

# ── 최적 파라미터 요약 표 ────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  최적 파라미터 요약")
print(f"{'=' * 80}")

for name in search_spaces:
    print(f"\n  [{name}]")
    for param, val in results[name].best_params_.items():
        print(f"    {param}: {val}")
