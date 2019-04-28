import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#This config fixes the truncated console writes. Makes things a lot easier
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)

dataLoc = 'D:\\Course Content\\python-for-data-science-and-ml-bootcamp\\13-Logistic-Regression\\'

#['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country', 'Timestamp', 'Clicked on Ad']
ad_df = pd.read_csv(dataLoc + 'advertising.csv')

#print(ad_df.describe())
#print(ad_df.info())
#print(ad_df.head(10))

#Do the most basic thing possible to see how predictive we are
ad_df_x = ad_df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
ad_df_y = ad_df['Clicked on Ad']

ad_df_trainx, ad_df_testx, ad_df_trainy, ad_df_testy = train_test_split(ad_df_x, ad_df_y, test_size=0.3, random_state=101)

lr = LogisticRegression()
lr.fit(X=ad_df_trainx, y=ad_df_trainy)

predictions = lr.predict(X=ad_df_testx)

print(confusion_matrix(y_true=ad_df_testy, y_pred=predictions))
print(classification_report(y_true=ad_df_testy, y_pred=predictions))

'''*********************************************************************************************

#We are given a train and a test set. Both will need to be cleaned up
titanic_train = pd.read_csv(dataLoc + 'titanic_train.csv')
titanic_test = pd.read_csv(dataLoc + 'titanic_test.csv')

print(f'Total number of rows in raw file: {titanic_train.shape[0]}')

#Look at the data incompleteness. There is a cool way of using the heatmap to see this
#sb.heatmap(data=titanic_train.isnull(), yticklabels=False, cbar=False)

#Countplot is quite useful. Automatically counts the number of occurences for the y axis value
#sb.countplot(data=titanic_train, x='Survived', hue = 'SibSp')

#Fixing the missing values for the Age column
#sb.boxplot(x = 'Pclass', y = 'Age', data = titanic_train)
#plt.show()

#!!!!!!!!!!!!CLEAN-UP the data !!!!!!!!!!!!!!!!!!!!!!!!!!!

#Get the average ages for each of the classes
avg_ages = pd.concat([titanic_train[['Pclass', 'Age']], titanic_test[['Pclass', 'Age']]], axis = 0).groupby('Pclass').mean()

#Function to set the age
def set_age(myrow, ages):
    if pd.isnull(myrow[0]):
        return round(ages.loc[myrow[1]])[0]
    else:
        return myrow[0]

#This syntax is interesting. It works because ....
#a) Axis = 1 tells the function to be applied to the entire row
#b) The default result_type is NONE and therefore it defaults to the return type of the function
titanic_train['Age'] = titanic_train[['Age', 'Pclass']].apply(lambda x: set_age(x, avg_ages), axis = 1)
titanic_test['Age'] = titanic_test[['Age', 'Pclass']].apply(lambda x: set_age(x, avg_ages), axis = 1)

#!!!!!!!!!!!! Deal with categorical variables !!!!!!!!!!!!!!!!!!!!!!!!!!!

#One-Hot Encode the Sex and Embarked variables
titanic_train = pd.concat([titanic_train, pd.get_dummies(data = titanic_train['Pclass'], drop_first = True), pd.get_dummies(data = titanic_train['Sex'], drop_first = True), pd.get_dummies(data = titanic_train['Embarked'], drop_first = True)], axis = 1)
titanic_test = pd.concat([titanic_test, pd.get_dummies(data = titanic_test['Pclass'], drop_first = True), pd.get_dummies(data = titanic_test['Sex'], drop_first = True), pd.get_dummies(data = titanic_test['Embarked'], drop_first = True)], axis = 1)

#Remove the variables that we will not use (are not going to be predictive)
titanic_train.drop(columns = ['PassengerId', 'Name', 'Sex', 'Cabin', 'Embarked', 'Ticket', 'Pclass'], axis = 1, inplace = True)
titanic_test.drop(columns = ['PassengerId', 'Name', 'Sex', 'Cabin', 'Embarked', 'Ticket', 'Pclass'], axis = 1, inplace = True)

#Remove all records that have NA, which is not too many
titanic_train.dropna(axis=0, inplace = True)
titanic_test.dropna(axis=0, inplace = True)

#Lets do our first model testing by splitting the training data

print(f'Total number of rows heading into training: {titanic_train.shape[0]}')

titanic_x = titanic_train.drop('Survived', axis = 1)
titanic_y = titanic_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(titanic_x, titanic_y, test_size = 0.3, random_state = 101)

lr = LogisticRegression()
lr.fit(X=x_train, y=y_train)

predictions = lr.predict(X=x_test)

comp_df = pd.concat([y_test.reset_index(), pd.DataFrame(predictions, columns=['Pred_Survived'])], axis=1)
misses = comp_df[comp_df['Survived'] == comp_df['Pred_Survived']]['index']

for i in misses:
    print(titanic_train.loc[i])

#print(confusion_matrix(y_true=y_test, y_pred=predictions))
#print(classification_report(y_true=y_test, y_pred=predictions))

'''
'''*********************************************************************************************

#My playing around with my own version of the sigmoid function and the s curve that it creates

myx = [-8,-6,-4,-2,0,2,4,6,8]
myy = []

for i, val in enumerate(myx):
    myy.insert(i, 1/(1+np.exp(-val)))

print(myy)

sb.lineplot(x = myx, y = myy)
plt.show()
'''