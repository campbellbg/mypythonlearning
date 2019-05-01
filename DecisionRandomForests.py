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

loans_orig = pd.read_csv(dataLoc + 'loan_data.csv')

#Look at the completeness of the data
#sb.heatmap(data=loans.isna())
#plt.show()

#Do we have a balanced classifier => Not really. 8045 Paid, 1533 Not Fully Paid.
#print(loans['not.fully.paid'].value_counts())

#This block of code way playing around with balancing the classifiers
#loans = loans_orig
loans = pd.concat([loans_orig[loans_orig['not.fully.paid'] == 0].loc[1:2000], loans_orig[loans_orig['not.fully.paid'] == 1]], axis=0)
loans.reset_index()

#Look at the categorical variables. One-Hot Encode these perhaps

#purpose => debt_consolidation, all_other, credit_card, home_improvement, small_business, major_purchase, educational
#print(loans['purpose'].value_counts())
#print(loans[['purpose', 'int.rate']].groupby('purpose').count())

#The independent variables dataframe
loans_dep = pd.concat([loans[['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec']], pd.get_dummies(data=loans['purpose'], drop_first=True)], axis=1)

#train test split
loans_x_train, loans_x_test, loans_y_train, loans_y_test = train_test_split(loans_dep, loans['not.fully.paid'], test_size=0.3, random_state=101)

'''
#Single decision tree. Capture performance for comparison with the Random Forest
my_dtree = DecisionTreeClassifier().fit(X=loans_x_train, y=loans_y_train)
loan_predictions = my_dtree.predict(X=loans_x_test)

'''

#Random Forest
my_rftree = RandomForestClassifier(n_estimators=500).fit(X=loans_x_train, y=loans_y_train)
loan_predictions = my_rftree.predict(X=loans_x_test)

#Confusion Matrix and Classification Report
print(confusion_matrix(y_true=loans_y_test, y_pred=loan_predictions))
print(classification_report(y_true=loans_y_test, y_pred=loan_predictions))

'''**********************************************************************************************************************

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