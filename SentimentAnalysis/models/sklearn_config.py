# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression

SVC = {'kernel': 'linear',  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
       'C': 1.0,
       'probability': True}

KNN = {'n_neighbors': 3}  # Number of neighbors to use

Logistic = {'solver': 'liblinear',
            'penalty': 'l2',  # 'l1' or 'l2'
            'C': 1.0}  # a positive float,Like in support vector machines, smaller values specify stronger
