#!/user/bin/env python3
# -*- coding: utf-8 -*-

###########
# summary
###########
"""Code file used to support Group Project Report and Notebook"""


###########
# Imports
###########
import numpy as np
import matplotlib.pyplot as plt
import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import adaBoost


############
# Functions
############




############
# Main
############
def main():
    """Code file control function"""
    print('Generating various Neural Network Results (allow some time)...')
    X, y = adaBoost.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    mi = 1000000
    learn_rates = [0.00005]  # ,0.0001, 0.00005, 0.000025, 0.00001]
    alphas = [0.00005]  #, 0.00005, 0.0001, 0.00025, 0.0005]
    results = []
    cols = [['score','training_score', 'activation', 'alpha', 'early_stopping'
            , 'learning_rate', 'learning_rate_init', 'max_iter'
            , 'momentum', 'solver', 'warm_start']]
    #     score activation  alpha early_stopping learning_rate learning_rate_init max_iter momentum solver warm_start
    # 0  0.9025   relu      0.00005   True      adaptive        0.00005  1000000   0.9   adam       True
    for lr in learn_rates:
        for alpha in alphas:
            nn = MLPClassifier(solver='adam'
                                , learning_rate='adaptive'
                                , activation='relu'
                                , hidden_layer_sizes=(128, 64, 64, 16, 8,)
                                , warm_start=True
                                , alpha=alpha
                                , learning_rate_init=lr
                                , random_state=13
                                , early_stopping=False
                                , momentum=0.99
                                , max_iter=mi)
            # print("NN config...\n", nn)
            clf = make_pipeline(StandardScaler(), nn)
            clf.fit(X_train, y_train)
            score = round(clf.score(X_test, y_test), 4)
            # grap a copy of all parameters used for this instance
            tr_score = round(clf.score(X_train, y_train))
            # print('weights: ', nn.coefs_)
            d = nn.get_params()
            row = [score, tr_score, d['activation'], d['alpha'], d['early_stopping']
                    , d['learning_rate'], d['learning_rate_init']
                    , d['max_iter'], d['momentum'], d['solver']
                    , d['warm_start']
                    ]
            # print(row)
            results.append(row)
    df = pd.DataFrame(results, columns=cols)
    print(df.head(30))
    # df.plot.bar(x='alpha', y='score')
    # plt.savefig('nn.png')

if __name__ == '__main__':
		main()
