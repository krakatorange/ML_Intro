'''
The following machine learning exercise covers the 
following subjects:

	- Installing the Python and SciPy platform.
	- Loading a dataset.
	- Summarizing a dataset.
	- Visualizing a dataset.
	- Evaluating machine learning algorithms.
	- Making predictions using machine learning algorithms.

Files:

	- ML_HotScript_01.py
	- iris.data
	
Python version:

	- Python 3.7.2
	
Packages used:

	- SciPy is a collection of numerical algorithms and 
	  domain-specific toolboxes.
	- Numpy is the fundamental package for numerical
	  computation.
	- Matplotlib for data visualization.
	- Pandas for data extraction and preparation.
	- Sklearn for working with classical ML algorithms.

Packages kept in mind for future consideration:

	- Tensorflow for Deep Learning.
	- Theano is also for Deep Learning.
	- Seaborn is another data visualization library.
	
Sources:

	- Project: machinelearningmastery.com/machine-learning-in-python-step-by-step
	- Data: archive.ics.uci.edu/ml/datasets/Iris, (University of California, Irvine)
'''

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# --------------------------------------------------------------------------------
# Loading the Data
# --------------------------------------------------------------------------------

# Load dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv('./iris.data', names=names)


# --------------------------------------------------------------------------------
# Summarizing the Dataset
# --------------------------------------------------------------------------------

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('class').size())


# --------------------------------------------------------------------------------
# Data Visualization
# --------------------------------------------------------------------------------

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# histograms
''' It looks like perhaps two of the input variables
	have a Gaussian distribution. '''
#dataset.hist()
#plt.show()

# scatter plot matrix
''' Note the diagonal grouping of some pairs of attributes.
	This suggests a high correlation and a predictable
	relationship. '''
#scatter_matrix(dataset)
#plt.show()


# --------------------------------------------------------------------------------
# Evaluating ML Algorithms
# --------------------------------------------------------------------------------

'''
Steps:

	- Separate out a validation dataset.
	- Set-up the test harness to use 10-fold cross validation.
	- Build 5 different models to predict species from flower measurements.
	- Select the best model.
	
Algorithms Evaluated:

	- Logistic Regression (LR)
	- Linear Discriminant Analysis (LDA)
	- K-Nearest Neighbors (KNN).
	- Classification and Regression Trees (CART).
	- Gaussian Naive Bayes (NB).
	- Support Vector Machines (SVM).
'''

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric (Test Harness)
''' The 'accuracy' metric is a ratio of the number of
	correctly predicted instances divided by the total
	number of instances in the dataset. '''
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
''' Warning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
	Limited Memory Broyden–Fletcher–Goldfarb–Shanno Algorithm - iterative method for solving
	unconstrained nonlinear optimization problems.
	Error: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.'
	Error occurs if max_iter exceeds 121. '''
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=121)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
# For this example, the SVM had a 99.167% confidence deeming it the winning model.
''' The objective of the Support Vector Machine algorithm is to find a hyperplane
in an N-dimensional space, where N is the number of features, that distinctly classifies
the data points. It also provides significant accuracy with less computational power
and is widely used in classification objectives. '''
	
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
svm = SVC(gamma='auto')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Conclusion:
''' The final accuracy score confirms good results with the 150 vector point dataset
	at a 93.333% confidence after using the SVM model.

					precision   recall  f1-score   support

	Iris-setosa     1.00     	1.00    1.00       7
	Iris-versicolor 1.00    	0.83    0.91       12
	Iris-virginica  0.85        1.00    0.92       11

	weighted avg    0.94        0.93    0.93       30
'''