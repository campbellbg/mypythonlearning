
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#This config fixes the truncated console writes. Makes things a lot easier
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)



''' USA Housing Data. Not real data
#***********************************************************************************************************************

#USA housing data
mydf = pd.read_csv('D:\\Course Content\\python-for-data-science-and-ml-bootcamp\\11-Linear-Regression\\USA_Housing.csv')
#mydf.info()

#print(mydf.corr()) #This is nice. A correlation matrix
#sb.heatmap(mydf.corr(), annot = True)
#sb.pairplot(mydf)
#sb.distplot(mydf['Price'])
#plt.show() #Need this in order for the chart to show within pycharm. No idea how seaborn and matplot are linked

mydfx = mydf[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']] #remove the dependent variable and the address
mydfy = mydf['Price']

mydfx_train, mydfx_test, mydfy_train, mydfy_test = train_test_split(mydfx, mydfy, test_size = 0.3, random_state = 101)

lm = LinearRegression() #instantiate an object
lm.fit(X = mydfx_train, y = mydfy_train) #fit (train model)

#print(f'Intercept = {lm.intercept_}')
#print(pd.DataFrame(data = lm.coef_, index = mydfx_train.columns, columns = ['Coeff'])) #Put the co-efficients into a dataframe for readibility

predictions = lm.predict(mydfx_test) #will give me back an array of my dependent variable predictions

#sb.distplot(mydfy_test - predictions) #Subtraction of the two aways works by index. Plot the residual distribution
#plt.show()

print(f'The mean abs error is {metrics.mean_absolute_error(y_true = mydfy_test, y_pred = predictions)}')
print(f'The mean squared error is {metrics.mean_squared_error(y_true = mydfy_test, y_pred = predictions)}')
print(f'The root mean squared error is {np.sqrt(metrics.mean_squared_error(y_true = mydfy_test, y_pred = predictions))}')
'''

''' Playing around with some of the syntax from the initial videos
#***********************************************************************************************************************

mymodel = LinearRegression(normalize=True)
print(mymodel) #must have an internal ToString method that allows this object to be printed out

#Create a bogus set of x and values. Uses a short-hand assignment notation
x, y = np.arange(10).reshape(5, 2), range(5)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 23) #random_state provides the same seed and therefore consistent results of the split

print(x_train)
print('------')
print(x_test)
print('------')
print(y_train)
print('------')
print(y_test)
print('######################')

#train (fit) the model
mymodel.fit(X = x_train, y = y_train)

#predict my values and store the result
predictions = mymodel.predict(x_test)

sum_squares = 0

for i, prediction in enumerate(predictions):
    print(f'X = {x_test[i]}, Actual Y = {y_test[i]}, Predicted Y = {prediction}')
    sum_squares += (prediction - y_test[i]) ** 2

print(sum_squares)

'''