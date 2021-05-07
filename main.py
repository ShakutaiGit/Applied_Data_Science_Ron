# This is a sample Python script.
from sklearn.model_selection import train_test_split
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sqlite3
from datetime import datetime
from statistics import mean

import shap
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from numpy.core import std
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from Pre_Processing import pre_processing
# Press the green button in the gutter to run the script.
def random_forest(result,data_in):
    X_train, X_test, y_train, y_test = train_test_split(data_in, result, test_size=0.2, random_state=0)
    # creating a RF classifier
    model = RandomForestClassifier(random_state=1)
    RandomForest_param = {
                        'bootstrap': [True, False],
                        'max_depth': [4,10, 30],
                        'n_estimators': [200, 400, 600]
    }
    search = GridSearchCV(estimator=model, param_grid=RandomForest_param, scoring='accuracy', n_jobs=1, cv=5,
                          refit=True, ).fit(X_train, y_train)

    result = search.predict(X_test)
    result_printer(y_test, result, "Random Forest Classifier",search)
    plotter_function(search.best_estimator_,X_train)


# def Logistic_Regression(data_in):
#     y = data_in["outcome"]
#     X = data_in.drop("outcome", axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     regr = LogisticRegression(solver='lbfgs', max_iter=1000)
#     param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'penalty':['l2']}
#     grid = GridSearchCV(regr, param_grid,  cv=6, scoring="accuracy").fit(X_train, y_train)
#     best_estimator = grid.best_estimator_.fit(X_train, y_train)
#     prediction_result = best_estimator.predict(X_test)
#     result_printer(y_test,prediction_result,"Logistic Regression")
#
#
# def GaussianNaiveBayes(data_in):
#     y = data_in["outcome"]
#     X = data_in.drop("outcome", axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     model = GaussianNB()
#     param_grid = {'n_init': [2, 5, 10], 'max_iter': [150, 300, 450, 600], 'algorithm': ["auto", "full", "elkan"]}
#     grid = GridSearchCV(model, param_grid, refit=True, cv=6).fit(X_train, y_train)
#     best_estimator = grid.best_estimator_.fit(X_train, y_train)
#     prediction_result = best_estimator.predict(X_test)
#     result_printer(y_test, prediction_result, "guassion model")
#
def MLP_Classifier(result,data_in):

    X_train, X_test, y_train, y_test = train_test_split(data_in, result, test_size=0.2, random_state=0)
    param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.005, 0.05],
    'learning_rate': ['constant','adaptive'],
        }
    model = MLPClassifier(max_iter=3500)
    grid = GridSearchCV(model, param_grid, refit=True, cv=5).fit(X_train, y_train)

    prediction_result = grid.predict(X_test)
    result_printer(y_test, prediction_result, "MLP_Classifier ",grid)

def K_Nearest_Neighbours_model(result,data_in):

    X_train, X_test, y_train, y_test = train_test_split(data_in, result, test_size=0.2, random_state=0)
    param_grid = {
        'leaf_size': [1, 5, 10,15]
        , 'n_neighbors': [ 3, 6, 9,12,15]
        ,'p':[1,2]
        ,'algorithm':['auto','ball_tree','kd_tree']}
    model = KNeighborsClassifier()
    grid = GridSearchCV(model, param_grid, refit=True, cv=5).fit(X_train, y_train)
    prediction_result = grid.predict(X_test)
    result_printer(y_test, prediction_result, "K_Nearest_Neighbours ",grid)

def svm_model(result,data_in):
    X_train, X_test, y_train, y_test = train_test_split(data_in, result, test_size=0.2, random_state=0)
    param_grid = {'C': [0.1, 1,0.01,10], 'gamma': [1, 0.1, 0.01, 0.05], 'kernel': ['linear', 'rbf', 'sigmoid']}
    model = SVC()
    grid = GridSearchCV(model, param_grid, refit=True, cv=6).fit(X_train, y_train)
    prediction_result = grid.predict(X_test)
    result_printer(y_test, prediction_result, "Svm model",grid)


def result_printer(y_test,y_prediction,model_name,model):
    print("{} model metrics results:".format(model_name))
    #y_normalize = list(zip(*y_test))[1]
    y_prediction= np.round(np.array(y_prediction))
    y_normalize = np.round(np.array(y_test.array))
    print("best_score:",model.best_score_)
    print("accuracy_score : ", metrics.accuracy_score(y_true=y_normalize, y_pred=y_prediction))
    # print("f1 score : ", metrics.f1_score(y_true=y_normalize, y_pred=y_prediction,average='micro'))
    # print("precision score :",metrics.precision_score(y_true=y_normalize,y_pred=y_prediction,average='macro'))
    # print("recall score :", metrics.recall_score(y_true=y_normalize, y_pred=y_prediction,average=None))
    print()
    print()
    print("classification report")
    target_names = ['class 0', 'class 1']
    print(metrics.classification_report(y_true=y_normalize,y_pred= y_prediction, target_names=target_names, zero_division=1))
    print(" confusion_matrix")
    print(metrics.confusion_matrix(y_normalize, y_prediction, ))
    print("normalized confusion_matrix")
    print(metrics.confusion_matrix(y_normalize, y_prediction, normalize='true'))
    print()
    print()

def plotter_function(model,X_train):
    #ploting Variable Importance Plot — Global Interpretability
    features = X_train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def createNewFeature(df):
    df["ratio_RBC_platelets"] = df["Red blood Cells"] / df["Platelets"]
    df["ratio_RDW_Platelets"] = df["Red blood cell distribution width (RDW)"] / df["Platelets"]
    df["ratio_Hemoglobin_Platelets"] = df["Hemoglobin"] / df["Platelets"]
    df["ratio_Hematocrit_Hemoglobin"] = df["Hematocrit"] / df["Hemoglobin"]
    df["ratio_Basophils_Leukocytes"] = df["Basophils"] / df["Leukocytes"]

def normlize_the_data(data_in):
    result = data_in["SARS-Cov-2 exam result"]
    X = data_in.drop("SARS-Cov-2 exam result", axis=1)
########### normlize data values################
    x = X.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return result, df

if __name__ == '__main__':
    path = r"Data/dataset/Bloot_Test_dataset/dataset.xlsx"
    pp = pre_processing(path, 0.95)
    clean_df = pp.clean_data()
    createNewFeature(clean_df)
    result, df = normlize_the_data(clean_df)
    MLP_Classifier(result,df)
    K_Nearest_Neighbours_model(result,df)
    #GaussianNaiveBayes(data_in)
    svm_model(result,df)
    random_forest(result,df)
    #Logistic_Regression(data_in)






