"""
EDA 통계 검정 실험
- 이탈/유지 그룹별 피처 분포 비교 (박스플롯)
- Mann-Whitney U 검정 (비모수 검정)
- 이상치 분석
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
import data_loader

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

# ── 설정 ─────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "matches"
OUTPUT_DIR = Path(__file__).parent / "results" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2015-05-25"
END_DATE = "2016-10-23"
ACTIVATION_PERIOD = 7
CHURN_OBSERVATION_PERIOD = 7

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

features = list(FEATURE_KO.keys())
churned = result_df[result_df[churn_col] == 0]
retained = result_df[result_df[churn_col] == 1]

print(f"이탈: {len(churned):,}명  |  유지: {len(retained):,}명")

# ── 1. 기술 통계량 ───────────────────────────────────────────
print(f"\n{'=' * 90}")
print(f"  1. 그룹별 기술 통계량")
print(f"{'=' * 90}")

desc_rows = []
for f in features:
    desc_rows.append({
        "피처": FEATURE_KO[f],
        "이탈 평균": churned[f].mean(),
        "이탈 중앙값": churned[f].median(),
        "이탈 표준편차": churned[f].std(),
        "유지 평균": retained[f].mean(),
        "유지 중앙값": retained[f].median(),
        "유지 표준편차": retained[f].std(),
    })

desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False, float_format="%.2f"))

# ── 2. Mann-Whitney U 검정 ───────────────────────────────────
print(f"\n{'=' * 90}")
print(f"  2. Mann-Whitney U 검정 (이탈 vs 유지)")
print(f"{'=' * 90}")
print(f"     귀무가설: 두 그룹의 분포가 동일하다")
print(f"     유의수준: 0.05 (Bonferroni 보정 후 {0.05 / len(features):.4f})\n")

bonferroni_alpha = 0.05 / len(features)

test_rows = []
for f in features:
    stat, p = stats.mannwhitneyu(churned[f], retained[f], alternative="two-sided")
    # effect size (rank-biserial correlation)
    n1, n2 = len(churned), len(retained)
    r = 1 - (2 * stat) / (n1 * n2)
    test_rows.append({
        "피처": FEATURE_KO[f],
        "U 통계량": stat,
        "p-value": p,
        "효과크기 (r)": abs(r),
        "유의": "***" if p < bonferroni_alpha else ("*" if p < 0.05 else "ns"),
        "방향": "이탈 > 유지" if churned[f].median() > retained[f].median() else "유지 > 이탈",
    })

test_df = pd.DataFrame(test_rows).sort_values("p-value")
print(f"{'피처':<16} {'p-value':>12} {'효과크기(r)':>12} {'유의':>6} {'방향':<14}")
print(f"{'-' * 62}")
for _, row in test_df.iterrows():
    print(f"{row['피처']:<16} {row['p-value']:>12.2e} {row['효과크기 (r)']:>12.4f} {row['유의']:>6} {row['방향']:<14}")

sig_count = (test_df["유의"] == "***").sum()
print(f"\nBonferroni 보정 후 유의한 피처: {sig_count}/{len(features)}개")

# ── 3. 박스플롯 (이탈 vs 유지) ───────────────────────────────
print(f"\n박스플롯 생성 중...")

n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()

for i, f in enumerate(features):
    ax = axes[i]
    data = [churned[f].values, retained[f].values]
    bp = ax.boxplot(data, labels=["이탈", "유지"], patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor("#EF553B")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#636EFA")
    bp["boxes"][1].set_alpha(0.6)
    ax.set_title(FEATURE_KO[f], fontsize=12, fontweight="bold")

    # 유의성 표시
    row = test_df[test_df["피처"] == FEATURE_KO[f]].iloc[0]
    if row["유의"] == "***":
        ax.text(0.95, 0.95, "***", transform=ax.transAxes, ha="right", va="top",
                fontsize=14, color="red", fontweight="bold")

# 남는 칸 숨김
for j in range(len(features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("이탈 vs 유지 그룹별 피처 분포 비교", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "boxplot_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {OUTPUT_DIR / 'boxplot_comparison.png'}")

# ── 4. 이상치 분석 (IQR 기준) ────────────────────────────────
print(f"\n{'=' * 90}")
print(f"  4. 이상치 분석 (IQR 기준)")
print(f"{'=' * 90}")

outlier_rows = []
for f in features:
    q1 = result_df[f].quantile(0.25)
    q3 = result_df[f].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((result_df[f] < lower) | (result_df[f] > upper)).sum()
    outlier_rows.append({
        "피처": FEATURE_KO[f],
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "하한": lower,
        "상한": upper,
        "이상치 수": n_outliers,
        "이상치 비율": n_outliers / len(result_df),
    })

outlier_df = pd.DataFrame(outlier_rows).sort_values("이상치 비율", ascending=False)
print(f"{'피처':<16} {'이상치 수':>10} {'비율':>8} {'하한':>10} {'상한':>10}")
print(f"{'-' * 56}")
for _, row in outlier_df.iterrows():
    print(f"{row['피처']:<16} {row['이상치 수']:>10,} {row['이상치 비율']:>8.1%} {row['하한']:>10.2f} {row['상한']:>10.2f}")

print(f"\n모든 결과가 {OUTPUT_DIR}/ 에 저장되었습니다.")
