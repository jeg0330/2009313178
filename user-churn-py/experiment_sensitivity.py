"""
AP/COP 민감도 분석 실험
- AP={3,5,7,10,14}, COP={3,5,7,10,14} 조합별 성능 변화
- Random Forest 기준 (최고 성능 모델)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
import data_loader

warnings.filterwarnings("ignore")

# ── 설정 ─────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "matches"
START_DATE = "2015-05-25"
END_DATE = "2016-10-23"
RANDOM_STATE = 42
K_FOLDS = 5

AP_VALUES = [3, 5, 7, 10, 14]
COP_VALUES = [3, 5, 7, 10, 14]

# ── 데이터 로드 ──────────────────────────────────────────────
print("데이터 로딩 중...")
raw_df = data_loader.load_parquet(str(DATA_DIR), START_DATE, END_DATE)
print(f"원본 레코드 수: {len(raw_df):,}")

cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── 실험 ─────────────────────────────────────────────────────
print(f"\nAP x COP 민감도 분석 (Random Forest, {K_FOLDS}-fold CV)")
print(f"AP: {AP_VALUES}")
print(f"COP: {COP_VALUES}")
print(f"총 {len(AP_VALUES) * len(COP_VALUES)}개 조합\n")

rows = []
for ap in AP_VALUES:
    for cop in COP_VALUES:
        churn_col = f"ap_{ap}d_and_cop_{cop}d"
        result_df = data_loader.filter_df(raw_df, ap, cop, churn_col)

        X = result_df.drop(churn_col, axis=1)
        y = np.ravel(result_df[churn_col])

        n_users = len(y)
        churn_rate = (y == 0).mean()

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=RANDOM_STATE,
        )

        acc_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
        auc_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
        f1_scores = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro")

        rows.append({
            "AP": ap,
            "COP": cop,
            "유저 수": n_users,
            "이탈 비율": churn_rate,
            "Accuracy": acc_scores.mean(),
            "AUC-ROC": auc_scores.mean(),
            "F1 (macro)": f1_scores.mean(),
        })

        print(f"  AP={ap:2d}, COP={cop:2d}  |  유저={n_users:>6,}  이탈={churn_rate:.1%}  "
              f"Acc={acc_scores.mean():.4f}  AUC={auc_scores.mean():.4f}  F1={f1_scores.mean():.4f}")

# ── 결과 테이블 ──────────────────────────────────────────────
df = pd.DataFrame(rows)

print(f"\n{'=' * 80}")
print(f"  AUC-ROC 히트맵 (행: AP, 열: COP)")
print(f"{'=' * 80}")
pivot_auc = df.pivot(index="AP", columns="COP", values="AUC-ROC")
print(pivot_auc.round(4).to_string())

print(f"\n{'=' * 80}")
print(f"  Accuracy 히트맵 (행: AP, 열: COP)")
print(f"{'=' * 80}")
pivot_acc = df.pivot(index="AP", columns="COP", values="Accuracy")
print(pivot_acc.round(4).to_string())

print(f"\n{'=' * 80}")
print(f"  이탈 비율 히트맵 (행: AP, 열: COP)")
print(f"{'=' * 80}")
pivot_churn = df.pivot(index="AP", columns="COP", values="이탈 비율")
print((pivot_churn * 100).round(1).to_string())

print(f"\n{'=' * 80}")
print(f"  유저 수 히트맵 (행: AP, 열: COP)")
print(f"{'=' * 80}")
pivot_users = df.pivot(index="AP", columns="COP", values="유저 수")
print(pivot_users.to_string())

# ── 최적 조합 ────────────────────────────────────────────────
best = df.loc[df["AUC-ROC"].idxmax()]
print(f"\n최고 AUC-ROC 조합: AP={int(best['AP'])}, COP={int(best['COP'])} → AUC={best['AUC-ROC']:.4f}")
print(f"  (유저 수: {int(best['유저 수']):,}, 이탈 비율: {best['이탈 비율']:.1%})")
