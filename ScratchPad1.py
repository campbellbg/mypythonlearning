
def isEven(innum):
    return True if innum % 2 == 0 else False #Simpler syntax for an if/else. Kind of like a tenerary operator

#Run this block when this file is executed direct i.e. as a the main call

if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    from math import modf

'''

    print('This section covers off the exercises for the pandas section of the course')

    #Index(['Id', 'EmployeeName', 'JobTitle', 'BasePay', 'OvertimePay', 'OtherPay', 'Benefits', 'TotalPay', 'TotalPayBenefits', 'Year', 'Notes', 'Agency', 'Status'],

    mysrc = 'D:\\Data\\'

    mydf = pd.read_csv(filepath_or_buffer=mysrc + 'salaries.csv', delimiter=',')

    print(f'The average base pay is {mydf["BasePay"].mean()}')
    print(f'The highest ovetime pay is {mydf["OvertimePay"].max()}')

    print(mydf[mydf['EmployeeName'] == 'JOSEPH DRISCOLL'][['JobTitle', 'TotalPayBenefits']])

    print(f"The highest paid person is {mydf.sort_values(by = ['TotalPayBenefits', 'TotalPay', 'BasePay', 'Id'], ascending = False).head(1)[['EmployeeName', 'TotalPayBenefits']]}")
    print(f"The lowest paid person is {mydf.sort_values(by=['TotalPayBenefits', 'TotalPay', 'BasePay', 'Id'], ascending=True).head(1)[['EmployeeName', 'TotalPayBenefits']]}")

    print(mydf[['Year', 'BasePay']].groupby(by = 'Year').mean().sort_values(by = 'Year'))

    print(f'There are {mydf["JobTitle"].nunique()} unique job titles')

    topjobs = mydf[['JobTitle', 'Id']].groupby(by = 'JobTitle').count()
    print(topjobs.sort_values(by = 'Id', ascending=False).head(5))

    print(topjobs[topjobs['Id'] == 1]['Id'].sum())

    print(mydf['EmployeeName'][mydf['JobTitle'].apply(lambda x: x.upper().find('CHIEF') != -1)].nunique())

    mydf['JobTitleLen'] = mydf['JobTitle'].apply(lambda x: len(x))

    print(mydf[['JobTitle', 'JobTitleLen']].head(4))

    print(mydf[['BasePay', 'JobTitleLen']].corr(method = 'pearson'))

'''

'''
    print('Read in a CSV file, manipulate some data and then write back the file')

    mysrc = 'D:\\Data\\'

    mydf = pd.read_csv(filepath_or_buffer = mysrc + 'usfailedbankslist.csv', delimiter = ',')

    mydf['Closing Month'] = mydf['Closing Date'].apply(lambda x: x[x.find('-') + 1:])

    mydf1 = mydf.groupby(by = 'Closing Month').count()[['Closing Date']].sort_values(by = 'Closing Date', ascending = False)
    mydf1.rename(columns = {'Closing Date':'Count'}, inplace=True) #Rename the columns

    mydf1.to_csv(path_or_buf= mysrc + 'banksbymonth.csv')

'''
'''
    mydf = pd.DataFrame({'col1': [1,2,3,4], 'col2':[444.4,555.43,66.3,444.4], 'col3':['abc','def','ghi','xyz']})

    print(mydf)

    #sort the dataframe
    mydf.sort_values(by = ['col2', 'col1'], inplace = True)
    mydf.reset_index(inplace = True)
    mydf.drop('index', axis = 1, inplace = True)
    print(mydf)

    #Pivot Tables
    print('This is just to play, I can''t imagine that these are going to be of that much use')

    mydf1 = mydf.pivot_table(values = 'col2', index = ['col1'], columns = ['col3'])
    mydf1.fillna(value = 0, inplace = True)
    print(mydf1)

'''

'''

    print('Have a look at the unique values within the dataframe')

    print(mydf['col1'].unique()) #provides a numpy array of the distinct values
    print(f'There are {mydf["col1"].nunique()} unique values in this column') #nunique counts the unique value. Same as finding the length of the array

    print(mydf.groupby('col2').count())
    print(mydf['col2'].value_counts())

    print('The apply method allows custom functions to be executed on series (row and columns?)')

    print(mydf['col2'].apply(lambda x: x * 2)) #Use apply on a column
    print(mydf.loc[1].apply(lambda x: x * 2)) #Use apply on a row

    #use apply to help to filter out based on a more complex condition? Like I was trying to do before
    print(mydf[mydf['col2'].apply(lambda x: modf(x)[1] % 2 == 0)])

    mydf['mod'] = mydf['col2'].apply(lambda x: modf(x)[1] % 2 == 0)
    print(mydf)

'''
'''

    print('Look at combining dataframes using Merge, Concatenation etc. This is essentially the replication of SQL join operations')

    mydf1 = pd.DataFrame({'A': [1, 2, 3, 4, 5, 99], 'B': ['A', 'B', 'C', 'D', 'E', 'F'], 'C': [10, 20, 30, 40, 50, 1000]})
    mydf2 = pd.DataFrame({'A': [6, 7, 8, 9, 10], 'B': ['A', 'B', 'C', 'D', 'E'], 'C': [60, 70, 80, 90, 100]})

    mydf3 = pd.concat([mydf1, mydf2], axis = 0) #Default is to operate like a union, which makes sense. Axis = 1 will try operate more like a join
    mydf3.reset_index(inplace=True)

    print(mydf3)

    print('Merge dataframes together, which is like a join')
    mydf4 = pd.merge(left = mydf1, right = mydf2, how = 'left', on = 'B') # How = Inner, Outer, Left, Right
    print(mydf4)

    print('Note that there is a join method for your dataframe but this always joins on the index rater than a specified column. Might not be very useful')
    print(mydf1.join(mydf2, lsuffix= 'df1', rsuffix = 'df2'))

'''
'''
    print('Having a play with the GroupBy function of pandas. Expect this to be fairly simple')

    mydict = {'Company':['G','G','M','M','F','F'], 'Person':['Sam','Charlie','Amy','Van','Carl','Sarah'], 'Sales': [200,120,340,124,243,350]}
    mydf = pd.DataFrame(mydict)

    print(mydf.describe()) #desribe function is useful.

    print(type(mydf.groupby(by = 'Company'))) #a groupby simply returns a groupby object. You have to perform an action on it i.e. a sum ...

    print(mydf.groupby(by='Company').sum())
    print(mydf.groupby(by='Company').count())

    print('Because dataframes are returned for each of these then all other operations i.e. conditionals, can be performed on these ')
    perssales = mydf.groupby(by='Person').sum()
    
    print(perssales[perssales > 200])

'''

'''
    
    print('Having a play with how to deal with missing data .. ')

    mydict = {'A': [1, 2, np.nan, 4], 'B': [4, np.nan, np.nan, 6], 'C': [1, 2, 3, np.nan]} #Use a dictionary to build a Dataframe. A very direct method
    mydf = pd.DataFrame(mydict)

    #DROPNA
    print(mydf)
    print(mydf.dropna()) #Drop where there is atleast 1 NA in the row
    print(mydf.dropna(axis = 1))  #Drop where ther is atleast 1 NA in the column
    print(mydf.dropna(axis = 0, thresh = 2)) #Drop where there is atleast 2 NA in the row

    #FILLNA
    print('Fill in the NA values using statics OR by using properties of the existing column i.e. mean, max ...')

    mydf['A'].fillna(value = mydf['A'].mean(), inplace = True) #Fill in column 'A' with the mean
    mydf['B'].fillna(value=mydf['B'].max(), inplace=True) #Fill in column 'B' with the max
    mydf['C'].fillna(value=-99, inplace=True)  # Fill in column 'C' with a static dummy value

    print(mydf)

'''
'''

    print('Have a play with multi level indexes. Provides a hierarchy in the index i.e. with the concept of a group of rows')

    mymd = pd.MultiIndex.from_tuples([('G1', 1), ('G1', 2), ('G1', 3), ('G2', 1), ('G2', 2), ('G2', 3)]) #multiindex with a grouping as a parent

    mydf = pd.DataFrame(data=myarr, index= mymd, columns=['A', 'B', 'C', 'D'])
    
    print(mydf)

    print('You can specify the names of the indexes, which is useful for references data particular using the cross-section (xs) method')
    mydf.index.names = ['GROUP', 'NUM']
    print(mydf.index.names)

    print('The cross section method can target certain levels in the multi-index hierarchy, as below')
    print(mydf.xs(key='G2', level=0))
    print(mydf.xs(key = 1, level = 1))

'''
'''

    np.random.seed(101) #set a common seed to keep the random numbers consistent with the demo

    myarr = np.random.randn(5, 4) #create my data

    mydf = pd.DataFrame(data = myarr, index = [i for i in range(1,6)], columns= ['A','B','C','D'])

    print('have a play with conditional selection. Returns df''s or series of booleans, much like numpy')

    print(mydf > 0) #This conditional checks all values are provides a dataframe of booleans
    print(mydf['A'] > 0) #This conditional checks the values in column 'A' and returns a series of booleans
    print(mydf[mydf['A'] > 0]) #passing the series of booleans limits the rows, kind of like a where clause

    print('It is possible to have multiple conditions by using the & and | characters. Not the traditional python and / or')

    print(mydf[(mydf['A'] > 0) & (mydf['B'] > 0)]) #You need to put these conditions in parentheses seperaeted by the & / |

'''

'''
    print(mydf)

    print('Selecting back columns is fairly standard, using the [] notation')

    print(type(mydf['A'])) #If you just select one column then you receive a series
    print(type(mydf[['A', 'B']])) #If you select multiple columns then you receive a daatframe

    print('You can add and remove columns from a dataframe, as you would expect')

    mydf['Z'] = 1 #Add my assigning directly to a new label
    mydf['X'] = 2

    print(mydf)

    mydf.drop(['X'], axis = 1, inplace = True) #to drop a column. Axis = 1 represents columns, Axis = 0 refers to rows. You must specify inplace = True for this to impact df
    mydf.drop([2], axis=0, inplace = True)

    print(mydf)

    print('Selecting rows it not as simple as a []. You need to use the loc keyword, with []')

    print(mydf.loc[3]) #Note the strange use of the []. Also note that this also returns a series

    print('loc is also used for selectivity within the dataframe')

    print(mydf.loc[:, 'B':]) #Note the use of ':' for slicing. Seems to work fine
    print(mydf.loc[[1, 5], ['A', 'B']]) #Or you can just use more explicit lists to get the data

'''

'''
    print('Create a few different series using different methods ...')
    mylab = labels = ['a', 'b', 'c', 'd']
    ser1 = pd.Series(data = [1, 2, 3, 4], index = mylab) #Use a python list
    ser2 = pd.Series(data = np.arange(1, 5), index = mylab) #Use a numpy array
    ser3 = pd.Series({'a': 1, 'b': 2, 'c': 3, 'e': 4})

    print(ser1)
    print(ser2)
    print(ser3)

    print(ser1 + ser2 + ser3) #you can sum them all together and it will 'join' on the common indexes to get the outcome. Like an inner join

'''

'''

    myarr = np.array([1, 2, 3, 4, 5])
    myarr = myarr * 2 #broadcasts to all elements in the array
    print(myarr)

    print('There are many universal np functions that are essentially static functions, called direct from the NP class.')

    print(np.sqrt(myarr))
    print(np.abs(myarr))
    print(np.max(myarr))

    myarr = np.arange(0, 11, 2)
    print(myarr[2:]) #slicing works exactly the same

    myarr1 = myarr[2:] #This will be a reference to the original array it DOES NOT take a copy
    myarr1[1] = -1

    print(myarr1)
    print(myarr) #Will see that the original array has changed as well

    myarr2 = np.linspace(0, 5, 20)
    myarr3 = myarr2.copy() #Take a copy of the array rather than maintaining a reference

    myarr3[2:4] = 99.99 #This is known as broadcasting. Set a slice of the array values to a figure

    print(myarr2)
    print(myarr3)

    print('Getting values from a matrix (2d array)')
    myarr = np.random.randint(0, 10, 25)
    myarr = myarr.reshape((5,5))

    print(myarr)

    print(myarr[0]) #the first row
    print(myarr[2][2]) #the second index from the second row
    print(myarr[2, 2]) #a different syntax to give the same as the above

    print('slice the matrix using the comma slice notation. Same as normal slicing, just thought of in 2d')

    print(myarr[:2, :2])
    print(myarr[:, 3:])

    print('Array conditional selection is pretty cool. In essence you can pass a conditional statement, which returns a boolean array')

    myarr = np.arange(0, 11)
    print(myarr % 2 == 0) # This returns an array of booleans based on the conditional evaluation

    print(myarr[myarr % 2 == 0]) #Can pass the condition directly into the selection and it will return only true elements
    print(myarr[np.array([True, False, False, False, False, False, False, False, False, False, True])]) #Prove how it works by forcing in my own boolean numpy array
'''

'''

    #You can cast a normal python list to a numpy array as follows
    myarr = np.array([1,2,3,4])
    print(myarr)
    print(myarr.reshape((2,2))) #perform a re-shape
    
    myarr = np.array([[1,2,3],[4,5,6], [7,8,9]])
    print(myarr)

    #arange is like range except that it generates a numpy array. Does not seem to be able to create a matrix
    print(np.arange(0, 10))

    #numpy zeroes and ones are quick ways to intansiate the array. Interesting to see how useul these might be
    print(np.zeros(5))
    print(np.zeros((5, 3)))

    #ones is similar to zeroes.
    a = np.ones((2,2))
    print(type(a[0][0] + 1))

    #linspace (assume that this stands for linear space, provides a vector of points between to numbers).
    print(np.linspace(start=0, stop= 5, num=20))
    print(np.linspace(start = 0, stop = -5, num = 20)) #Seems to work for negative direction as well

    #identity matrix (do some reading). A square matrix that carries 1's down the diagonal
    print(np.eye(3))

    #Random generation of numbers
    print(np.random.rand(5))  # Random floats between 0 and 1. Parameter tells me how many
    print(np.random.rand(5, 5)) #Random floats between 0 and 1. Parameter tells me how many

    print(np.random.randn(5)) #Random numbers from a normal distribution that centers on zero
    print(np.random.randn(3, 3))  # Random numbers from a normal distribution that centers on zero

    print(np.random.randint(low = 1, high = 50, size = 20)) #Get 20 random integers between 1 and 50

    #re-shape a matrix (must have the correct number of values to fill the dimension)
    myarr = np.arange(0, 25)
    print(myarr.reshape((5, 5)))

    #Maximum and Minimum Values
    print(myarr.max())
    print(myarr.min())
    print(myarr[myarr.argmax()]) #argmax returns the position of the maximum value. This shows using it to get the max in a different manner
    print(myarr[myarr.argmin()])

'''

'''
    #List comprehension example
    mylist = [x for x in range(0, 50) if x % 2 == 0]
    print(mylist)

    #Filter function .... filter(lambda, iteratable)
    mylist1 = list(filter(lambda x: x % 2 == 0, range(0, 50)))
    print(mylist1)

    #Simpler syntax for an if/else. Kind of like a tenerary operator
    mylist2 = list(filter(lambda x: isEven(x), range(0, 50)))
    print(mylist2)

'''

