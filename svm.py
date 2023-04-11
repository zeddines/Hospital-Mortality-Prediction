import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split


def read_into_list():
		"""returns csv file into vector of columns and matrix of features

		Returns:
		[list]: list of column names as string
		[list]: larger matrix of features
		"""
		# read in data01.csv into lists
		columns =  []
		x_features = []
		with open('data01.csv', 'r') as csv:
				csv_lines = csv.readlines()
				columns = csv_lines[0].split(',')
				for line in csv_lines[1:]:
						row = line.split(',')
						x_features.append(row)
		return columns, x_features


def plot_feature(X, y, feature_name, i):
		"""plot feature and outcome

		Args:
				x (matrix): features(Xi, Xj)
				y (vector):	outcome
		"""
		if not X.shape[0] == y.shape[0]:
			print(f"cannot print {feature_name} as x and y are not the same length")
		# yx = np.c_[y, x]
		# # print("x, y shapes: ", x.shape, y.shape)
		# # print('yx...', yx)
		# xn = [list(row) for row in yx if row[0] <= 0]
		# xp = [list(row) for row in yx if row[0] > 0]	
		plt.subplot(6, 4, i)
		plt.scatter(X, y, s=1)
		plt.yticks([-1, 0, 1, 2], ['', 'alive', 'dead', ''])
		plt.title(f"Scatter of '{feature_name}' values")
		plt.xlabel(feature_name)
		return

def scatter_features(X_a, X_d, col_list):
		"""Scatter plot pairings of features

		Args:
				X_a ([pd.dataframe]): pandas dataframe of outcome=0
				X_d ([pd.dataframe]): pandas dataframe of outcome=1
				col_list: list of column names for outer loop
		"""
		
		# plot from age onwards
		for x_feature in col_list:
			for compare_to in col_list:
				if x_feature == compare_to:
					continue
				# plot outcome=0=alive
				plt.scatter(X_a[x_feature], X_a[compare_to], s=5, c='green', marker='+', label='Alive')
				# plot records of deceased persons outcome=1
				plt.scatter(X_d[x_feature], X_d[compare_to], s=3, c='red', marker='o', label='Died')
				plt.title(f"{x_feature} and {compare_to}")
				plt.xlabel(x_feature)
				plt.ylabel(compare_to)
				plt.legend()
				plt.tight_layout()
				plt.show()
				# figname = f"./ScatterPlots/{x_feature}_{compare_to}.jpg"
				# plt.savefig(figname)


def get_binary_cols(df, column_names, num_cols, print_cols=True):
		binary_cols = []
		for i in range(num_cols):
				# plot_feature(x = X[:, i], y=y, feature_name=X_cols[i], i=i+1)
				col_name = column_names[i]
				min_v = df[col_name].min()
				max_v = df[col_name].max()
				diff = max_v - min_v
				print(f"{col_name}:\t\tmin={min_v}, max={max_v}, diff={diff}")
				if diff <= 2:
					binary_cols.append(col_name)
		if print_cols:
			print('Binary columns...')
			for col in binary_cols:
				print(col)
		return binary_cols



def find_p_value(df):
		d0 = df.to_numpy()
		# rows, cols = d0.shape
		# print(f"d0 has {rows} rows and {cols} columns")
		y = d0[:,2]
		X = d0[:,3:]
		X_cols = df.columns.tolist()
		# print('shape of X: ', X.shape)
		# print('shape of y: ', y.shape)
		# plt.figure(1)
		
		# find p-values
		mod = sm.OLS(y, X)
		fii = mod.fit()
		p_vals = fii.summary2().tables[1][:]
		print(p_vals)
		# for c in X_cols[3:]:
		# 	print(c)


def get_best_columns(df:pd.DataFrame):
		X = df.iloc[:,3:]
		y = df.iloc[:,2]
		best_features = SelectKBest(score_func=chi2, k=10)
		fit = best_features.fit(X, y)
		dfscores = pd.DataFrame(fit.scores_)
		dfcolumns = pd.DataFrame(X.columns)
		feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
		feature_scores.columns = ['xFeature', 'Score']
		print('Select KBest Features...')
		print(feature_scores.nlargest(15, 'Score'))


def get_important_cols(df:pd.DataFrame):
		X = df.iloc[:,3:]
		y = df.iloc[:,2]
		model = ExtraTreesClassifier()
		model.fit(X, y)
		important_features = pd.Series(model.feature_importances_, index=X.columns)
		print('ExtraTreeClassifier Important Features found')
		print(important_features)
		important_features.nlargest(15).plot(kind='barh')
		plt.xlabel('feature score')
		plt.title('Top 15 Features found by ExtraTreesClassifier')
		plt.show()


# predict and report results of classifiers
def report_results(clf_list, clf_names, X, y, cr=False):
		"""report results

		Args:
				clf_list: list of classifiers
				clf_names: list of classifier names
				X: data to use for prediction
				y: data to compare results
		"""
		to = len(clf_list)
		for i in range(to):
				clf = clf_list[i]
				lbl = clf_names[i]
				pred = clf.predict(X)
				score = clf.score(X, y)
				mse = round(mean_squared_error(y, pred), 3)
				acc = round(accuracy_score(y, pred), 3)
				print(f"{lbl} score:\t{round(score, 2)}\tmse: {mse}\t Accuracy: {acc}")
				if cr:
					print(classification_report(y,pred, target_names=['survived','mortality']
																			, zero_division=1))
				if lbl == 'VC':
					print(confusion_matrix(y, pred))


# create and fit classifiers and ensembles
def train_test_data(X, y):
		"""Train, test and report accuracy models

		Args:
				X ([list]): features
				y ([list]): target class
		"""
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		clf_dt = DecisionTreeClassifier(max_depth=4)
		clf_knn = KNeighborsClassifier(n_neighbors=3)
		clf_svc = svm.SVC(gamma=0.1, kernel='rbf', probability=True)
		clf_rf = RandomForestClassifier(n_estimators=400, max_depth=2)
		eclf = VotingClassifier(
			estimators=[('dt', clf_dt), ('knn', clf_knn), ('svc', clf_svc), ('rf', clf_rf)]
			, voting='hard'
		)
		clf_dt.fit(X_train, y_train)
		clf_knn.fit(X_train, y_train)
		clf_svc.fit(X_train, y_train)
		clf_rf.fit(X_train, y_train)
		eclf.fit(X_train, y_train)

		print('\nTraining tests (testing on same data fitted to)')
		clfs = [clf_dt, clf_knn, clf_svc, clf_rf, eclf]
		lbls = ['DT', 'KNN', 'SVC', 'RF', 'VC']
		report_results(clfs, lbls, X_train, y_train)

		print('\nTesting Results (with unseen test data)')
		report_results(clfs, lbls, X_test, y_test, True)
		# print('SVC support: ', clf_svc.support_)
		# print('SVC support_vectors length: ', len(clf_svc.support_vectors_))



def main():
		# import data into a pandas dataframe
		df = pd.read_csv('data01.csv', sep=',')
		df.replace('NA', np.nan, inplace=True)
		columns = df.columns.tolist()
		# print(df.dtypes)
		print('df indexes: \n', df.index)		
		df.fillna(df.median(), inplace=True)
		df_alive = df.where(df['outcome'] == 0)
		df_dead = df.where(df['outcome'] == 1)
		df_alive.dropna(subset=['outcome'], inplace=True)
		df_dead.dropna(subset=['outcome'], inplace=True)
		# find the best columns...
		# get_best_columns(df)
		# get_important_cols(df)
		# find_p_value(df)
		# scatter_features(df_dead, df_alive, key_cols)
		X = df[['NT-proBNP', 'Creatine kinase', 'Urine output'
	      , 'Urea nitrogen', 'Platelets', 'Leucocyte', 'Lymphocyte'
				, 'Anion gap', 'Lactic acid', 'Bicarbonate', 'Blood calcium'
				, 'Blood sodium', 'Basophils', 'PT', 'heart rate'
				, 'Systolic blood pressure', 'glucose', 'Diastolic blood pressure'
				, 'Respiratory rate', 'age', 'BMI', 'INR', 'Renal failure'
				, 'temperature', 'Neutrophils', 'PH', 'SP O2']]
		y = df.iloc[:,2]
		train_test_data(X, y)

		

#######
# Main
#######
if __name__ == '__main__':
		main()
