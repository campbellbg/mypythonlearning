import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.datasets import load_breast_cancer #This is an inbuilt dataset that comes with sklearn

from sklearn.svm import SVC #Support Vector Classifer
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#This config fixes the truncated console writes. Makes things a lot easier
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)

#The Project

#Get the iris dataset from sb
#['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = sb.load_dataset(name='iris')

#print(iris_data.columns)
#print(iris_data.describe())

#sb.pairplot(data=iris_data)
#plt.show()

#Train Test Split
x_train, x_test, y_train, y_test = train_test_split(iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], iris_data['species'], test_size=0.2, random_state=101)

'''
#instantiate and fit the SVM model, with default hyperparameters
my_svc = SVC().fit(X=x_train, y=y_train)

predictions = my_svc.predict(X=x_test)

print(confusion_matrix(y_true=y_test, y_pred=predictions))
print('/n')
print(classification_report(y_true=y_test, y_pred=predictions))
'''
#Even though the above model work really well, do a grid search anyway just for my learning

my_params = {'C': [0.1, 0.2, 0.3, 1, 2, 10], 'gamma': [0.1, 0.2, 0.3, 1, 2, 10]}
my_grid = GridSearchCV(estimator=SVC(), param_grid=my_params).fit(X=x_train, y=y_train)#Instantiate and fit GridSearch

print(my_grid.best_params_)

predictions = my_grid.predict(X=x_test)

print(confusion_matrix(y_true=y_test, y_pred=predictions))
print('/n')
print(classification_report(y_true=y_test, y_pred=predictions))


''' !!!!!!!!!!!!!!!! This is the lecture work for SVM's

bc_data = load_breast_cancer()

bc_df_feat = pd.DataFrame(data = bc_data['data'], columns = bc_data['feature_names'])
#bc_df_tgt = pd.DataFrame(data = bc_data['target'], columns = ['is_benign'])

#Train Test split, nothing special here
x_train, x_test, y_train, y_test = train_test_split(bc_df_feat, bc_data['target'], test_size=0.3, random_state=111)

#!!!!!!!!! Model without GridSearch on the Hyperparameters. Performs terribly

#Instantiate and fit the model
#svc_model = SVC().fit(X=x_train, y=y_train)

#Look at the predictions. Will be able to use confusion matrix and classification report
#predictions = svc_model.predict(X=x_test)

#print(confusion_matrix(y_true=y_test, y_pred=predictions))
#print('/n')
#print(classification_report(y_true=y_test, y_pred=predictions))

#Use a Grid Search to look for the Hyperparameters

my_params = {'C':[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001, 10]}
my_grid = GridSearchCV(estimator=SVC(), param_grid=my_params, verbose=0) #Put verbose at 0 so that it doesn't spool to much to my output window

my_grid.fit(X=x_train, y=y_train)

#print(my_grid.best_params_)
#print(my_grid.best_estimator_)

predictions = my_grid.predict(X=x_test)

print(confusion_matrix(y_true=y_test, y_pred=predictions))
print('/n')
print(classification_report(y_true=y_test, y_pred=predictions))

'''
