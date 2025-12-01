#SVM Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Logistic Regression Imports
from sklearn import linear_model

#KNN Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def train_svm(df, df_xcols, df_ycol, kernel_type):
    x_svm = df[df_xcols]
    y_svm = df[df_ycol]

    X_train, X_test, y_train, y_test = train_test_split(x_svm, y_svm, test_size=0.3) # 70% training and 30% test

    clf = SVC(kernel=kernel_type)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    #Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #Model Precision
    print("Precision:",metrics.precision_score(y_test, y_pred))

    #Model Recall
    print("Recall:",metrics.recall_score(y_test, y_pred))

    #Model F1 Score
    print("F1 Score: ", metrics.f1_score(y_test, y_pred))

    return clf

def train_logr(df, feature_cols, df_ycol):
    x_logr = df[feature_cols]
    y_logr = df[df_ycol]
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(x_logr, y_logr, test_size=0.3)

    logr = linear_model.LogisticRegression(max_iter = 1000)
    logr.fit(X_train, y_train.values.ravel())
    y_pred = logr.predict(X_test)

    #Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #Model Precision
    print("Precision:",metrics.precision_score(y_test, y_pred))

    #Model Recall
    print("Recall:",metrics.recall_score(y_test, y_pred))

    #Model F1 Score
    print("F1 Score: ", metrics.f1_score(y_test, y_pred))

    return logr

def train_KNN(df, feature_cols, df_ycol, neighbors):

    # Create feature and target arrays
    x_KNN = df[feature_cols]
    y_KNN = df[df_ycol]

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(x_KNN, y_KNN, test_size = 0.3)

    knn = KNeighborsClassifier(n_neighbors=neighbors)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    #Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #Model Precision
    print("Precision:",metrics.precision_score(y_test, y_pred))

    #Model Recall
    print("Recall:",metrics.recall_score(y_test, y_pred))

    #Model F1 Score
    print("F1 Score: ", metrics.f1_score(y_test, y_pred))

    return knn