# This second scratchpad is for my refresher following a break from working with pandas and numpy

import numpy as np

#Vector = 1 dimensional

vect1 = np.arange(0, 5)
vect2 = np.array([1, 2, 3, 4, 5])

#print(vect1)
#print(vect2)

#Matrix = 2 dimensional

#Use a python list of lists to generate the matrix
mtx1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#Use arange to get to a vector and then reshape it as a matrix
mtx2 = np.arange(0, 25).reshape((5, 5))

#Slicing

#Uses the [row, column] or [row], [col] format for selection. Slicing can be applied to this, similar to a python list

#print(mtx1[1:, 2:])

#Broadcast to an entire slice. Does this update the object or a reference to it?

#print(mtx1)
#mtx1[1] = 9 #Casts the entire selection (slice) to a value
#print(mtx1)

#Numpy functions on arrays i.e. max
#print(mtx1.max())
#print(mtx1.max(axis=0)) #Maximum for each column
#print(mtx1.max(axis=1)) #Maximum for each row
#print(mtx1.max()) #Maximum for the entire matrix, including rows and columns

#Conditional selection on a numpy array is pretty cool. Pass in a boolean list or an expression that returns a boolean list
#print(mtx1[mtx1 > 2])

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Now have a play with Pandas

import pandas as pd

#A series is an individual column in a pandas dataframe. Not often a need to deal with series but it helps to understand the fundamentals
ser1 = pd.Series(data = [1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])
ser2 = pd.Series(data = np.arange(0, 4), index = ['a', 'b', 'c', 'd'])

#print(ser2)

#A dataframe is a collection of series (one series is a column).
#df1 = pd.DataFrame(np.arange(0, 25).reshape(5,5), columns = ['a', 'b', 'c', 'd', 'e'])

#Selection and slicing uses a similar syntax except there is a need to use the LOC keywords if rows and columns are needed
#print(df1.loc[2:,'b':'c'])

#unique and nunique
#print(df1.sort_values(by = 'a', ascending=False))

#Read in a dataframe from file and have a play with it to answer certain questions
mysrc = 'D:\\Data\\'
mydf = pd.read_csv(filepath_or_buffer = mysrc + 'Salaries.csv', delimiter = ',')

#Average pay by year
#print(mydf[['Year', 'BasePay']].groupby('Year').sum())

#Find the job with the highest pay
#print(mydf[['JobTitle', 'BasePay']].groupby('JobTitle').sum().sort_values(by='BasePay', ascending=False).head(2))

#How many different titles have chief in them? Allow for capitalisation

def myupper(x):
    return x.upper()

mydf1 = mydf['JobTitle'].apply(func=myupper)

print(f"There are {mydf1.nunique()} unique job titles ..... AND ")

print(mydf1[mydf1.apply(lambda x: x.find('CHIEF')== 12)])

#missing data. isnull, dropna, fillna

