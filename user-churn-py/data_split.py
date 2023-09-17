from sklearn.model_selection import train_test_split

def data_split(df, t_col, test_size, random_state=123456):
    X = df.drop(t_col, axis='columns')
    y = df[[t_col]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X, y, X_train, X_test, y_train, y_test

