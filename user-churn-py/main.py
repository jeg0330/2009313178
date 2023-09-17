## data load
import dataframe_loader

result_df = dataframe_loader.load_df("bulkmatches1-20000.json")

## data split
import data_split

X, y, X_train, X_test, y_train, y_test = data_split.data_split(result_df, 'played_next_7_days', 0.3,
                                                               random_state=123456)
## data preprocessing
import data_visualization

data_visualization.visualize_column_counts(result_df, 'played_next_7_days')
data_visualization.visualize_correlation_matrix(result_df, method='pearson')
data_visualization.visualize_scatter_matrix(['score', 'points', 'degree', 'flair'], 'played_next_7_days', result_df)

## model training
import model_training

## 1. KNN
model_training.knn_classifier(X_train, y_train, X_test, y_test)
model_training.random_forest_classifier(X_train, X_test, y_train, y_test)
model_training.naive_bayes_classifier(X_train, y_train, X_test, y_test)
model_training.support_vector_classifier(X_train, y_train, X_test, y_test)
