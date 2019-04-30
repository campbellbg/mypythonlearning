import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#This config fixes the truncated console writes. Makes things a lot easier
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)

#['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
#['not.fully.paid']
dataLoc = 'D:\\Course Content\\python-for-data-science-and-ml-bootcamp\\15-Decision-Trees-and-Random-Forests\\'

loans = pd.read_csv(dataLoc + 'loan_data.csv')

#Look at the completeness of the data

#Look at the categorical variables. One-Hot Encode these perhaps

#Remove features that aren't really useful i.e. not very predictive ....

#train test split

#Random Forest

#Confusion Matrix and Classification Report

'''
#['Kyphosis', 'Age', 'Number', 'Start']
my_df = pd.read_csv(dataLoc + 'kyphosis.csv')

#plt.figure(figsize=(10,10))
#sb.pairplot(data=my_df, hue='Kyphosis')
#plt.show()

#Train and Test Split
x_train, x_test, y_train, y_test = train_test_split(my_df[['Age', 'Number', 'Start']], my_df['Kyphosis'], test_size=0.3, random_state=101)

#Decision Tree
my_tree = DecisionTreeClassifier()
my_tree.fit(X=x_train, y=y_train)

my_predictions = my_tree.predict(X=x_test)

#Success Rate
print(np.mean(y_test == my_predictions))

print(confusion_matrix(y_true=y_test, y_pred=my_predictions))
print(classification_report(y_true=y_test, y_pred=my_predictions))

#Random Forest
my_forest = RandomForestClassifier(n_estimators=200)
my_forest.fit(X=x_train, y=y_train)

my_predictions = my_forest.predict(X=x_test)

print(confusion_matrix(y_true=y_test, y_pred=my_predictions))
print('/n')
print(classification_report(y_true=y_test, y_pred=my_predictions))
'''