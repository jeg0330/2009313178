"""
PR Curve + Threshold 분석 실험
- Precision-Recall Curve + Average Precision
- Threshold 변화에 따른 Precision/Recall trade-off
- Random Forest 기준
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import warnings
import data_loader

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

# ── 설정 ─────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "matches"
OUTPUT_DIR = Path(__file__).parent / "results" / "pr_threshold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2015-05-25"
END_DATE = "2016-10-23"
ACTIVATION_PERIOD = 7
CHURN_OBSERVATION_PERIOD = 7
RANDOM_STATE = 42

# ── 데이터 로드 + 모델 학습 ──────────────────────────────────
print("데이터 로딩 중...")
raw_df = data_loader.load_parquet(str(DATA_DIR), START_DATE, END_DATE)
churn_col = f"ap_{ACTIVATION_PERIOD}d_and_cop_{CHURN_OBSERVATION_PERIOD}d"
result_df = data_loader.filter_df(raw_df, ACTIVATION_PERIOD, CHURN_OBSERVATION_PERIOD, churn_col)

X = result_df.drop(churn_col, axis=1)
y = np.ravel(result_df[churn_col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y,
)

rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    class_weight="balanced", random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)

# 이탈(class=0) 확률
y_proba_churn = rf.predict_proba(X_test)[:, 0]
# PR/ROC 계산 시 positive class = 이탈(0)이므로 y 반전
y_test_churn = (y_test == 0).astype(int)

# ── 1. PR Curve + ROC Curve 나란히 ───────────────────────────
precision, recall, pr_thresholds = precision_recall_curve(y_test_churn, y_proba_churn)
ap = average_precision_score(y_test_churn, y_proba_churn)

fpr, tpr, roc_thresholds = roc_curve(y_test_churn, y_proba_churn)
auc = roc_auc_score(y_test_churn, y_proba_churn)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PR Curve
axes[0].plot(recall, precision, color="#EF553B", linewidth=2, label=f"AP = {ap:.4f}")
axes[0].axhline(y=y_test_churn.mean(), color="gray", linestyle="--", label=f"기준선 (이탈 비율 = {y_test_churn.mean():.2f})")
axes[0].set_xlabel("Recall (재현율)", fontsize=12)
axes[0].set_ylabel("Precision (정밀도)", fontsize=12)
axes[0].set_title("Precision-Recall Curve", fontsize=14)
axes[0].legend(fontsize=10)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].grid(alpha=0.3)

# ROC Curve
axes[1].plot(fpr, tpr, color="#636EFA", linewidth=2, label=f"AUC = {auc:.4f}")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="랜덤 기준선")
axes[1].set_xlabel("False Positive Rate", fontsize=12)
axes[1].set_ylabel("True Positive Rate", fontsize=12)
axes[1].set_title("ROC Curve", fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pr_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {OUTPUT_DIR / 'pr_roc_curve.png'}")
print(f"  Average Precision: {ap:.4f}")
print(f"  AUC-ROC: {auc:.4f}")

# ── 2. Threshold 분석 ────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  Threshold별 성능 변화")
print(f"{'=' * 80}")

thresholds_to_test = np.arange(0.1, 1.0, 0.05)
rows = []
for t in thresholds_to_test:
    y_pred = (y_proba_churn >= t).astype(int)
    if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
        continue
    rows.append({
        "Threshold": t,
        "Precision": precision_score(y_test_churn, y_pred),
        "Recall": recall_score(y_test_churn, y_pred),
        "F1": f1_score(y_test_churn, y_pred),
        "Accuracy": accuracy_score(y_test_churn, y_pred),
        "이탈 예측 수": y_pred.sum(),
        "이탈 예측 비율": y_pred.mean(),
    })

thresh_df = pd.DataFrame(rows)
print(thresh_df.to_string(index=False, float_format="%.4f"))

# ── 3. Threshold trade-off 그래프 ────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(thresh_df["Threshold"], thresh_df["Precision"], "o-", color="#EF553B", label="Precision", linewidth=2)
ax1.plot(thresh_df["Threshold"], thresh_df["Recall"], "s-", color="#636EFA", label="Recall", linewidth=2)
ax1.plot(thresh_df["Threshold"], thresh_df["F1"], "^-", color="#00CC96", label="F1", linewidth=2)
ax1.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="기본 threshold (0.5)")

# F1 최대 지점
best_f1_idx = thresh_df["F1"].idxmax()
best_t = thresh_df.loc[best_f1_idx, "Threshold"]
best_f1 = thresh_df.loc[best_f1_idx, "F1"]
ax1.axvline(x=best_t, color="#00CC96", linestyle=":", alpha=0.7)
ax1.annotate(f"최적 F1={best_f1:.4f}\n(t={best_t:.2f})",
             xy=(best_t, best_f1), xytext=(best_t + 0.08, best_f1 - 0.05),
             fontsize=10, arrowprops=dict(arrowstyle="->", color="#00CC96"))

ax1.set_xlabel("Threshold (이탈 판정 기준)", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_title("Threshold에 따른 Precision / Recall / F1 변화", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim([0.1, 0.9])
ax1.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "threshold_tradeoff.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  저장: {OUTPUT_DIR / 'threshold_tradeoff.png'}")

# ── 운영 시나리오 제안 ───────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  운영 시나리오별 Threshold 제안")
print(f"{'=' * 80}")

# Recall 90% 이상인 최대 Precision threshold
high_recall = thresh_df[thresh_df["Recall"] >= 0.9]
if not high_recall.empty:
    best_hr = high_recall.loc[high_recall["Precision"].idxmax()]
    print(f"\n  [이탈 유저 최대 포착] Recall >= 90%")
    print(f"    Threshold: {best_hr['Threshold']:.2f}")
    print(f"    Precision: {best_hr['Precision']:.4f}, Recall: {best_hr['Recall']:.4f}")
    print(f"    → 이탈 유저의 {best_hr['Recall']:.0%}를 잡지만, 예측 중 {1 - best_hr['Precision']:.0%}는 오탐")

# F1 최대
best_f1_row = thresh_df.loc[thresh_df["F1"].idxmax()]
print(f"\n  [균형 최적] F1 최대")
print(f"    Threshold: {best_f1_row['Threshold']:.2f}")
print(f"    Precision: {best_f1_row['Precision']:.4f}, Recall: {best_f1_row['Recall']:.4f}")

# Precision 80% 이상인 최대 Recall threshold
high_prec = thresh_df[thresh_df["Precision"] >= 0.8]
if not high_prec.empty:
    best_hp = high_prec.loc[high_prec["Recall"].idxmax()]
    print(f"\n  [정확한 타겟팅] Precision >= 80%")
    print(f"    Threshold: {best_hp['Threshold']:.2f}")
    print(f"    Precision: {best_hp['Precision']:.4f}, Recall: {best_hp['Recall']:.4f}")
    print(f"    → 이탈 예측의 {best_hp['Precision']:.0%}가 실제 이탈, 대신 {1 - best_hp['Recall']:.0%} 놓침")
