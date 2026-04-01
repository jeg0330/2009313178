import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def knn_classifier(X_train, y_train, X_test, y_test):
    neighbors = []
    cv_scores = []

    for k in range(1, 17, 1):
        neighbors.append(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, np.ravel(y_train))
        scores = cross_val_score(knn, X_train, np.ravel(y_train), cv=7, scoring='accuracy')
        cv_scores.append(scores.mean())

    # Misclassification error versus k
    MSE = [1 - x for x in cv_scores]

    optimal_k = neighbors[MSE.index(min(MSE))]

    # Evaluation

    model = KNeighborsClassifier(n_neighbors=optimal_k)
    model.fit(X_train, np.ravel(y_train))
    y_pred = np.ravel(model.predict(X_test))

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print('kNN Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%  |  AUC-ROC: {auc:.4f}  |  best k={optimal_k}')
    print('----------------------------------------------')
    return y_test, y_pred, model


def random_forest_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                   class_weight='balanced', random_state=42)
    model.fit(X_train, np.ravel(y_train))

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print('RF Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%  |  AUC-ROC: {auc:.4f}')
    print('----------------------------------------------')

    return y_test, y_pred, model


def naive_bayes_classifier(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, np.ravel(y_train))

    y_pred = gnb.predict(X_test)

    auc = roc_auc_score(y_test, gnb.predict_proba(X_test)[:, 1])
    print('Naive Bayes Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%  |  AUC-ROC: {auc:.4f}')
    print('----------------------------------------------')


def support_vector_classifier(X_train, y_train, X_test, y_test):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, np.ravel(y_train))

    y_pred = svclassifier.predict(X_test)

    auc = roc_auc_score(y_test, svclassifier.decision_function(X_test))
    print('SVM Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%  |  AUC-ROC: {auc:.4f}')
    print('----------------------------------------------')


def xgboost_classifier(X_train, X_test, y_train, y_test):
    from xgboost import XGBClassifier

    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=(np.ravel(y_train) == 0).sum() / (np.ravel(y_train) == 1).sum(),
                          eval_metric='logloss', random_state=42)
    model.fit(X_train, np.ravel(y_train))

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print('XGBoost Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%  |  AUC-ROC: {auc:.4f}')
    print('----------------------------------------------')

    return y_test, y_pred, model


def lightgbm_classifier(X_train, X_test, y_train, y_test):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           is_unbalance=True, random_state=42, verbose=-1)
    model.fit(X_train, np.ravel(y_train))

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print('LightGBM Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%  |  AUC-ROC: {auc:.4f}')
    print('----------------------------------------------')

    return y_test, y_pred, model


## cv_val
def cv_val(model, X, y, k):
    cv_scores = cross_val_score(model, X, np.ravel(y), cv=k, scoring='accuracy')
    print(cv_scores)
    print('avg_cv_scores: ', np.average(cv_scores))
