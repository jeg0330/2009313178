"""
k-Fold 교차 검증 실험
- 5개 모델에 대해 동일한 조건(5-fold stratified CV)으로 평가
- 보고서용 결과 표 출력: Accuracy, AUC-ROC, F1(macro) 의 평균 +/- 표준편차
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# ── 데이터 로드 ──────────────────────────────────────────────
print("데이터 로딩 중...")
raw_df = data_loader.load_parquet(str(DATA_DIR), START_DATE, END_DATE)
print(f"원본 레코드 수: {len(raw_df):,}")

churn_col = f"ap_{ACTIVATION_PERIOD}d_and_cop_{CHURN_OBSERVATION_PERIOD}d"
result_df = data_loader.filter_df(raw_df, ACTIVATION_PERIOD, CHURN_OBSERVATION_PERIOD, churn_col)

X = result_df.drop(churn_col, axis=1)
y = np.ravel(result_df[churn_col])

print(f"유저 수: {len(y):,}  |  이탈: {(y == 0).sum():,}  |  유지: {(y == 1).sum():,}  |  이탈 비율: {(y == 0).mean():.2%}")
print(f"피처 수: {X.shape[1]}")

# ── 모델 정의 ────────────────────────────────────────────────
models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("nb", GaussianNB()),
    ]),
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

# ── 교차 검증 ────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "roc_auc", "f1_macro"]

print(f"\n{'=' * 80}")
print(f"  {K_FOLDS}-Fold Stratified 교차 검증")
print(f"  AP={ACTIVATION_PERIOD}일, COP={CHURN_OBSERVATION_PERIOD}일, random_state={RANDOM_STATE}")
print(f"{'=' * 80}")

summary_rows = []
fold_detail = {}

for name, model in models.items():
    print(f"  {name} 학습 중...", end=" ", flush=True)
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    acc = results["test_accuracy"]
    auc = results["test_roc_auc"]
    f1 = results["test_f1_macro"]

    summary_rows.append({
        "모델": name,
        "Accuracy": f"{acc.mean():.4f} +/- {acc.std():.4f}",
        "AUC-ROC": f"{auc.mean():.4f} +/- {auc.std():.4f}",
        "F1 (macro)": f"{f1.mean():.4f} +/- {f1.std():.4f}",
        "_acc": acc.mean(),
        "_auc": auc.mean(),
        "_f1": f1.mean(),
    })

    fold_detail[name] = {"accuracy": acc, "auc_roc": auc, "f1_macro": f1}
    print(f"Acc={acc.mean():.4f}  AUC={auc.mean():.4f}  F1={f1.mean():.4f}")

# ── 결과 요약 표 ─────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
display_cols = ["모델", "Accuracy", "AUC-ROC", "F1 (macro)"]

print(f"\n{'=' * 80}")
print(f"  결과 요약")
print(f"{'=' * 80}")
print(summary_df[display_cols].to_string(index=False))

best_idx = summary_df["_auc"].idxmax()
best = summary_df.iloc[best_idx]
print(f"\n  >> 최고 AUC-ROC: {best['모델']} ({best['AUC-ROC']})")

# ── Fold별 상세 ──────────────────────────────────────────────
for metric_name, metric_key in [("Accuracy", "accuracy"), ("AUC-ROC", "auc_roc"), ("F1 (macro)", "f1_macro")]:
    print(f"\n{'=' * 80}")
    print(f"  Fold별 상세 — {metric_name}")
    print(f"{'=' * 80}")

    detail_rows = []
    for name in models:
        scores = fold_detail[name][metric_key]
        row = {"모델": name}
        for i, s in enumerate(scores):
            row[f"Fold {i + 1}"] = f"{s:.4f}"
        row["평균"] = f"{scores.mean():.4f}"
        row["std"] = f"{scores.std():.4f}"
        detail_rows.append(row)

    print(pd.DataFrame(detail_rows).to_string(index=False))
