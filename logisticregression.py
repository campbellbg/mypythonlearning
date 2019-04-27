import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#This config fixes the truncated console writes. Makes things a lot easier
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)

dataLoc = 'D:\\Course Content\\python-for-data-science-and-ml-bootcamp\\13-Logistic-Regression\\'

#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
titantic_train = pd.read_csv(dataLoc + 'titanic_train.csv') # nrows = 20) #playing around with nrows to improve the performance of the read
row_cnt = titantic_train.shape[0]

#titanic_test = pd.read_csv(dataLoc + 'titanic_test.csv')

#Look at the data incompleteness. There is a cool way of using the heatmap to see this
#sb.heatmap(data=titantic_train.isnull(), yticklabels=False, cbar=False)

#Countplot is quite useful. Automatically counts the number of occurences for the y axis value
#sb.countplot(data=titantic_train, x='Survived', hue = 'SibSp')

#Fixing the missing values for the Age column
#sb.boxplot(x = 'Pclass', y = 'Age', data = titantic_train)
#plt.show()

#!!!!!!!!!!!!CLEAN-UP the data !!!!!!!!!!!!!!!!!!!!!!!!!!!

#Get the average ages for each of the classes
avgages = titantic_train[['Pclass', 'Age']].groupby('Pclass').mean()

#Function to set the age
def set_age(myrow, ages):
    if pd.isnull(myrow[0]):
        return round(ages.loc[myrow[1]])
    else:
        return myrow[0]

#This syntax is interesting. It works because ....
#a) Axis = 1 tells the function to be applied to the entire row
#b) The default result_type is NONE and therefore it defaults to the return type of the function
titantic_train['Age'] = titantic_train[['Age', 'Pclass']].apply(lambda x: set_age(x, avgages), axis = 1)

#Remove all records that have NA, which is not too many
titantic_train.dropna(inplace = True)

print(f'During data cleanup we have dropped {row_cnt - titantic_train.shape[0]} rows')

#!!!!!!!!!!!!CLEAN-UP the data !!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!! Deal with categorical variables !!!!!!!!!!!!!!!!!!!!!!!!!!!

#One-Hot Encode the Sex and Embarked variables
titantic_train = pd.concat([titantic_train, pd.get_dummies(data = titantic_train['Sex'], drop_first = True), pd.get_dummies(data = titantic_train['Embarked'], drop_first = True)], axis = 1)

#Remove the variables that we will not use (are not going to be predictive)
titantic_train.drop(columns = ['PassengerId', 'Name', 'Sex', 'Cabin', 'Embarked', 'Ticket'], axis = 1, inplace = True)

print(titantic_train.head(2))

#@@@@@ WORTH SEEING IF CHANGING PCLASS TO ENCODED VALUES MAKES ANY DIFFERENCE TO THE ACCURACY

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