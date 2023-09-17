import seaborn as sns
import matplotlib.pyplot as plt


def visualize_column_counts(df, t_col, font_size=1.5):
    plt.clf()
    plt.figure(figsize=(12, 8))  # 히트맵의 크기를 조절합니다.

    sns.set(font_scale=font_size)
    fig1 = sns.countplot(x=t_col, data=df)
    fig1.set(ylabel='Count')
    for p in fig1.patches:
        height = p.get_height()
        fig1.text(p.get_x() + p.get_width() / 2, height + 3, height, ha='center', size=20)
    plt.show()


def visualize_correlation_matrix(df, method, font_size=1.5):
    plt.clf()
    plt.figure(figsize=(12, 8))  # 히트맵의 크기를 조절합니다.
    sns.set(font_scale=font_size)
    corr_matrix = df.corr(method)

    # 모든 값 표시
    sns.heatmap(corr_matrix, center=0, annot=True, fmt=".2f")

    plt.title('Correlation Map')
    plt.show()


def visualize_scatter_matrix(col_list, target_col, df, font_size=1.5):
    plt.clf()
    plt.figure(figsize=(12, 8))  # 히트맵의 크기를 조절합니다.

    sns.set(font_scale=font_size)
    sns.pairplot(data=df, vars=col_list, hue=target_col, palette='bright', markers='+')

    plt.show()
