import seaborn as sns
import matplotlib.pyplot as plt


def visualize_column_counts(df, t_col, font_size=1.5):
    plt.figure(figsize=(12, 8))

    sns.set(font_scale=font_size)
    fig1 = sns.countplot(x=t_col, data=df)
    fig1.set(ylabel='Count')
    for p in fig1.patches:
        height = p.get_height()
        fig1.text(p.get_x() + p.get_width() / 2, height + 3, height, ha='center', size=20)
    plt.show()


def visualize_correlation_matrix(df, method, font_size=1.5):
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=font_size)
    corr_matrix = df.corr(method=method, numeric_only=True)

    # 모든 값 표시
    sns.heatmap(corr_matrix, center=0, annot=True, fmt=".2f")

    plt.title('Correlation Map')
    plt.show()


def visualize_scatter_matrix(col_list, target_col, df, font_size=1.5):
    plt.figure(figsize=(12, 8))

    sns.set(font_scale=font_size)
    sns.pairplot(data=df, vars=col_list, hue=target_col, palette='bright', markers='+')

    plt.show()


def visualize_feature_importance(model, feature_names, font_size=1.5):
    plt.clf()
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=font_size)

    importances = model.feature_importances_
    indices = importances.argsort()

    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()
