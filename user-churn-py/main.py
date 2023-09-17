## data load
import data_loader

filename = "bulkmatches1-20000.json"
df = data_loader.load_df(filename)

### summary
print(df['date'].min())
print(df['date'].max())
print(df['date'].max() - df['date'].min())

## data filter
activation_period = 14
target_period = 3
target_column = 'target_value'
result_df = filter_df(df, activation_period, target_period, target_column)

X, y, X_train, X_test, y_train, y_test = data_loader.data_split(result_df, target_column, 0.3,
                                                                random_state=4)
## data visualization
import data_visualization

data_visualization.visualize_column_counts(result_df, target_column)
data_visualization.visualize_correlation_matrix(result_df, method='pearson')
# data_visualization.visualize_scatter_matrix(['score', 'points', 'degree', 'flair'], target_column, result_df)

## model training
import model_training

model_training.knn_classifier(X_train, y_train, X_test, y_test)
model_training.random_forest_classifier(X_train, X_test, y_train, y_test)
model_training.naive_bayes_classifier(X_train, y_train, X_test, y_test)
# model_training.support_vector_classifier(X_train, y_train, X_test, y_test)
