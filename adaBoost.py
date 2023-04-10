import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import zero_one_loss, accuracy_score, confusion_matrix



# load data from file
def load_data():
    """Loads initial csv file and retuns X, y"""
    df = pd.read_csv('data01.csv', sep=',')
    df.replace('NA', np.nan, inplace=True)
    columns = df.columns.tolist()
		# print('df indexes: \n', df.index)
    df.fillna(df.median(), inplace=True)
    X = df[['NT-proBNP', 'Creatine kinase', 'Urine output'
	      , 'Urea nitrogen', 'Platelets', 'Leucocyte', 'Lymphocyte'
				, 'Anion gap', 'Lactic acid', 'Bicarbonate', 'Blood calcium'
				, 'Blood sodium', 'Basophils', 'PT', 'heart rate'
				, 'Systolic blood pressure', 'glucose', 'Diastolic blood pressure'
				, 'Respiratory rate', 'age', 'BMI', 'INR', 'Renal failure'
				, 'temperature', 'Neutrophils', 'PH', 'SP O2']]
    y = df.iloc[:,2]
    return X, y


# returns two adaBoost models, one for each alogrithm of "SAMME" and "SAMME.R"
def trained_adaBoost_models(X, y, dt_stump, learn_rate, n):
    """creates and trains the two adaBoost models("SAMME" and "SAMME.R")
    
    Args:
        x (matrix): features
        y (vector): target
        dt_stump (DecisionTree): decisionTree with depth 1
        learn_rate (float): models learning rate
        n (int): number of iterations
    """
    ada = AdaBoostClassifier(estimator=dt_stump
                               , learning_rate=learn_rate
                               , n_estimators=n
                               , algorithm='SAMME'
                               )
    ada.fit(X, y)
    ada2 = AdaBoostClassifier(estimator=dt_stump
                              , learning_rate=learn_rate
                              , n_estimators=n
                              , algorithm="SAMME.R")
    ada2.fit(X, y)
    return ada, ada2


# predit and report model error
def predict_report(adaB_model, X, y):
    """Makes prediction and returns error"""
    errs = []
    for pred in enumerate(adaB_model.staged_predict(X)):
        errs.append(zero_one_loss(y, pred[1]))
    return errs


# plot adaBoost training results
def plot_adaBoost_results(dt_1_err, dt_2_err, a1_errs, a1_tr_errs
                            , a2_errs, a2_tr_errs, n, fig_num):
    """creates a plot of adaBoost training/test prediction results"""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([1, n], [dt_1_err] * 2, "k-", label='DT d=1 Error')
    ax.plot([1, n], [dt_2_err] * 2, "k--", label='DT d=7 Error')
    colours = sns.color_palette("pastel")
    # discrete ada results
    ax.plot(np.arange(n) + 1, a1_errs, label='ada test err'
            , color=colours[0])
    ax.plot(np.arange(n) + 1, a1_tr_errs, label='ada train err'
            , color=colours[1])
    # real ada results
    ax.plot(np.arange(n) + 1, a2_errs, label='ada2 test err'
            , color=colours[2])
    ax.plot(np.arange(n) + 1, a2_tr_errs, label='ada2 train err'
            , color=colours[4])
    ax.set_xlabel('Number of Weak Learners')
    ax.set_ylabel('Error Rate')
    ax.legend(loc='upper right')
    fig.suptitle('DecisionTree and AdaBoost Training and Test Results')
    # plt.show()
    plt.savefig(f'./TrainTestPics/figure_adaBoost_{fig_num}.png')
    


# ref: https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html
def main():
    """main place to set up objects"""
    # get data
    X, y = load_data()
    y = y.to_numpy()
    X = X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = 0.5
    n = 300
    # first set up some weak learners
    stump = DecisionTreeClassifier(max_depth=1)  # , min_samples_leaf=1)
    stump.fit(X_train, y_train)
    stump_err = 1.0 - stump.score(X_test, y_test)
    dtc = DecisionTreeClassifier(max_depth=7)
    dtc.fit(X_train, y_train)
    dtc_err = 1.0 - dtc.score(X_test, y_test)
    # create two AdaBoost classifiers, for algos "SAMME" and "SAMME.R"
    ada, ada2 = trained_adaBoost_models(X_train, y_train, stump, lr, n)
    # get the errors for test (& also training) whiile predicting
    ada_errors = predict_report(ada, X_test, y_test)
    ada_train_err = predict_report(ada, X_train, y_train)
    ada2_err = predict_report(ada2, X_test, y_test)
    ada2_train_err = predict_report(ada2, X_train, y_train)
    # plot the results
    plot_adaBoost_results(stump_err, dtc_err
                            , ada_errors, ada_train_err
                            , ada2_err, ada2_train_err, n, fig_num='abisdt1a')




#######
# Main
#######
if __name__ == '__main__':
		main()