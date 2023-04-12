#!/user/bin/env python3
# -*- coding: utf-8 -*-

##############
# Summary
##############

# Script Author: Tanya
# Contributors: Tanya, Novin, Bertram

"""
Various Machine Learning models configured and trained to predict
the unseen test data
"""


##############
# Imports
##############

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, \
  auc, precision_recall_curve, average_precision_score, \
  ConfusionMatrixDisplay, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier
import preprocessing as pp
import feature_enginnering as fe



##############
# Functions
##############

# Authors: Tanya, Novin
def neural_net_config():
    """Configures (learned) parameters for scikit-learn's MLP classifier

    Parameters:
        - X (pandas Dataframe): Training set Features
        - y (pandas Series): Training set Labels/Targets
    
    Returns: MLPClassifier object trained on X and y
    """
    nn = MLPClassifier(
            solver='adam'
            , learning_rate='adaptive'
            , activation='tanh'
            , alpha=0.00005
            , early_stopping=False
            , learning_rate_init=0.00015
            , max_iter=3000
            , momentum=0.99
            , warm_start=True            
    )
    return nn
    


# Author: Tanya
# Configures a list of classifier models for training and testing
def configure_models():
    """Generates a list of ML models to predic hospital mortality

    Returns: List of configured models
    """
    # create the classifiers
    # neural network MLP
    nn = neural_net_config()
    # XG Boost Classifier
    xgb = XGBClassifier(alpha=0.7, max_depth=2, learning_rate=0.9
                        , n_estimators=25, gamma=0.1)
    # Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(loss='log_loss'
                                     , n_estimators=100
                                     , learning_rate=0.91
                                     , max_depth=2
                                     , random_state=43)
    dt = DecisionTreeClassifier(max_depth=4)
    knn = KNeighborsClassifier(n_neighbors=3)
    svc = svm.SVC(gamma=0.1, kernel='rbf', probability=True)
    rf = RandomForestClassifier(n_estimators=400, max_depth=2)
    vc = VotingClassifier(estimators=[
           ('nn', nn), ('xgb', xgb), ('gbc', gbc), ('dt', dt)
           , ('knn', knn), ('svc', svc), ('rf', rf)
    ], voting='hard')
    lbs = ['nn', 'xgb', 'gbc', 'dt', 'knn', 'svc', 'rf', 'vc']
    clfs = [nn, xgb, gbc, dt, knn, svc, rf, vc]
    return lbs, clfs



def train_test_predict(X, y, X_tst, y_tst, lbls, models):
      """train test and predict each model then generatre results
      
      Parameters:
          - X (Dataframe): Features training set
          - y (Series): Target training labels
          - X_tst(Dataframe): Unseen features
          - y_txt(Series): True Targets
          - lbls (str): model names
          - models(objects): classifier models
      """
      num_models = len(models)
      for i in range(num_models):
          clf = models[i]
          clf.fit(X, y)
          # predict training and test
          train_pred = clf.predict(X)
          y_pred = clf.predict(X_tst)
          # get MSE for training and test
          tr_mse = round(mean_squared_error(y, train_pred), 3)
          mse = round(mean_squared_error(y_tst, y_pred), 3)
          # find accuracy score
          tr_acc = round(accuracy_score(y, train_pred), 3)
          acc = round(accuracy_score(y_tst, y_pred), 3)
          lbl = lbls[i]
          print(f"{lbl} training\tMSE:{tr_mse}\tacc: {tr_acc}")
          print(f"{lbl} testing\tMSE:{mse}\tacc:{acc}")
          print('Score', round(clf.score(X_tst, y_tst), 3))
          print()
       
    





##############
# Main
##############

# Author: Tanya
def main():
    """ function which controls code called """
    print("Mortality Prediction...")
    # have pre-processing module apply data cleansing
    pp_df = pp.initial_data_cleansing()
    # add features with feature engineering module
    X_train, X_test, y_train, y_test = fe.model_ready_datasets(pp_df)
    clf_lbls, clf_models = configure_models()
    train_test_predict(X_train, y_train, X_test, y_test, clf_lbls, clf_models)


    
    



# fist line of code exuted in preprocess.py if run as script
if __name__ == '__main__':
		main()

