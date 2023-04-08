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






# ref: https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html
def main():
    """main place to set up objects"""
    # get data
    X, y = load_data()
    y = y.to_numpy()
    # print(y)
    X = X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
	  # first set up some weak learners
    stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    stump.fit(X_train, y_train)
    stump_err = 1.0 - stump.score(X_test, y_test)
    dtc = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
    dtc.fit(X_train, y_train)
    dtc_err = 1.0 - dtc.score(X_test, y_test)
    lr = 0.5
    n = 750
    # create two AdaBoost classifiers, for algos "SAMME" and "SAMME.R"
    ada = AdaBoostClassifier(estimator=stump
                               , learning_rate=lr
                               , n_estimators=n
                               , algorithm='SAMME'
                               )
    ada.fit(X_train, y_train)
    ada2 = AdaBoostClassifier(estimator=stump
                              , learning_rate=lr
                              , n_estimators=n
                              , algorithm="SAMME.R")
    ada2.fit(X_train, y_train)
    # do test error first
    ada_errors = [] 
    for pred in enumerate(ada.staged_predict(X_test)):
        # print('pred', pred[1])
        # print('y_test', y_test)
        ada_errors.append(zero_one_loss(y_test, pred[1]))
    # now do training error
    ada_train_err = []
    for pred in enumerate(ada.staged_predict(X_train)):
        ada_train_err.append(zero_one_loss(y_train, pred[1]))
    ada2_err = []
    for pred in enumerate(ada2.staged_predict(X_test)):
        ada2_err.append(zero_one_loss(y_test, pred[1]))
    ada2_train_err = []
    for pred in enumerate(ada2.staged_predict(X_train)):
        ada2_train_err.append(zero_one_loss(y_train, pred[1]))
    
    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([1, n], [stump_err] * 2, "k-", label='Decision Stump Error')
    ax.plot([1, n], [dtc_err] * 2, "k--", label='Decision Tree Error')
    colours = sns.color_palette("pastel")
    # discrete ada results
    ax.plot(np.arange(n) + 1, ada_errors, label='ada test err'
            , color=colours[0])
    ax.plot(np.arange(n) + 1, ada_train_err, label='ada train err'
            , color=colours[1])
    # real ada results
    ax.plot(np.arange(n) + 1, ada2_err, label='ada2 test err'
            , color=colours[2])
    ax.plot(np.arange(n) + 1, ada2_train_err, label='ada2 train err'
            , color=colours[3])
    ax.set_xlabel('Number of Weak Learners')
    ax.set_ylabel('Error Rate')
    ax.legend()
    plt.show()
    



#######
# Main
#######
if __name__ == '__main__':
		main()