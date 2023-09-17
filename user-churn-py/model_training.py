import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report


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

    print('kNN Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print('Accuracy of diagnosis prediction using KNN = ', accuracy_score(y_pred, y_test) * 100)

    return y_test, y_pred, model


def random_forest_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=10, bootstrap=False)
    model.fit(X_train, np.ravel(y_train))

    y_pred = model.predict(X_test)

    print('RF Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print('Accuracy of diagnosis prediction using Naive Bayes = ', accuracy_score(y_pred, y_test) * 100)

    return y_test, y_pred, model


def naive_bayes_classifier(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, np.ravel(y_train))

    y_pred = gnb.predict(X_test)

    print('Naive Bayes Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print('Accuracy of diagnosis prediction using Naive Bayes = ', accuracy_score(y_pred, y_test) * 100)


def support_vector_classifier(X_train, y_train, X_test, y_test):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, np.ravel(y_train))

    y_pred = svclassifier.predict(X_test)

    print('SVM Classification Report : \n')
    print(classification_report(y_test, y_pred))
    print('Accuracy of diagnosis prediction using SVM = ', accuracy_score(y_pred, y_test) * 100)


## cv_val
def cv_val(model, X, y, k):
    cv_scores = cross_val_score(model, X, np.ravel(y), cv=k, scoring='accuracy')
    print(cv_scores)
    print('avg_cv_scores: ', np.average(cv_scores))
