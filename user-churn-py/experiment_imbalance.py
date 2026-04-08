"""
클래스 불균형 처리 비교 실험
- 처리 없음 / class_weight / SMOTE 3가지 전략 비교
- 5개 모델 x 3가지 전략 = 15개 조합
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
churn_col = f"ap_{ACTIVATION_PERIOD}d_and_cop_{CHURN_OBSERVATION_PERIOD}d"
result_df = data_loader.filter_df(raw_df, ACTIVATION_PERIOD, CHURN_OBSERVATION_PERIOD, churn_col)

X = result_df.drop(churn_col, axis=1)
y = np.ravel(result_df[churn_col])

print(f"유저 수: {len(y):,}  |  이탈: {(y == 0).sum():,} ({(y == 0).mean():.1%})  |  유지: {(y == 1).sum():,} ({(y == 1).mean():.1%})")

cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "roc_auc", "f1_macro", "recall_macro"]

pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

# ── 전략별 모델 정의 ─────────────────────────────────────────
strategies = {
    "처리 없음": {
        "KNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=7))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "Naive Bayes": Pipeline([("scaler", StandardScaler()), ("nb", GaussianNB())]),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbose=-1),
    },
    "가중치 조정": {
        "KNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE),
        "Naive Bayes": Pipeline([("scaler", StandardScaler()), ("nb", GaussianNB())]),  # NB는 가중치 미지원
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, is_unbalance=True, random_state=RANDOM_STATE, verbose=-1),
    },
    "SMOTE": {
        "KNN": ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=RANDOM_STATE)), ("knn", KNeighborsClassifier(n_neighbors=7))]),
        "Random Forest": ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE))]),
        "Naive Bayes": ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=RANDOM_STATE)), ("nb", GaussianNB())]),
        "XGBoost": ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=RANDOM_STATE))]),
        "LightGBM": ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("lgbm", LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbose=-1))]),
    },
}

# ── 실험 ─────────────────────────────────────────────────────
all_rows = []
model_names = ["KNN", "Random Forest", "Naive Bayes", "XGBoost", "LightGBM"]

for strategy_name, models in strategies.items():
    print(f"\n{'=' * 80}")
    print(f"  전략: {strategy_name}")
    print(f"{'=' * 80}")

    for model_name in model_names:
        model = models[model_name]
        print(f"  {model_name}...", end=" ", flush=True)

        results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

        acc = results["test_accuracy"].mean()
        auc = results["test_roc_auc"].mean()
        f1 = results["test_f1_macro"].mean()
        recall = results["test_recall_macro"].mean()

        all_rows.append({
            "전략": strategy_name,
            "모델": model_name,
            "Accuracy": acc,
            "AUC-ROC": auc,
            "F1 (macro)": f1,
            "Recall (macro)": recall,
        })
        print(f"Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  Recall={recall:.4f}")

# ── 결과 비교 ────────────────────────────────────────────────
df = pd.DataFrame(all_rows)

print(f"\n{'=' * 80}")
print(f"  전략별 AUC-ROC 비교")
print(f"{'=' * 80}")
pivot_auc = df.pivot(index="모델", columns="전략", values="AUC-ROC")
pivot_auc = pivot_auc[["처리 없음", "가중치 조정", "SMOTE"]]
print(pivot_auc.round(4).to_string())

print(f"\n{'=' * 80}")
print(f"  전략별 F1 (macro) 비교")
print(f"{'=' * 80}")
pivot_f1 = df.pivot(index="모델", columns="전략", values="F1 (macro)")
pivot_f1 = pivot_f1[["처리 없음", "가중치 조정", "SMOTE"]]
print(pivot_f1.round(4).to_string())

print(f"\n{'=' * 80}")
print(f"  전략별 Recall (macro) 비교")
print(f"{'=' * 80}")
pivot_recall = df.pivot(index="모델", columns="전략", values="Recall (macro)")
pivot_recall = pivot_recall[["처리 없음", "가중치 조정", "SMOTE"]]
print(pivot_recall.round(4).to_string())

# ── 최적 조합 ────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  최적 조합 (AUC-ROC 기준)")
print(f"{'=' * 80}")
best = df.loc[df["AUC-ROC"].idxmax()]
print(f"  {best['모델']} + {best['전략']} → AUC={best['AUC-ROC']:.4f}")
