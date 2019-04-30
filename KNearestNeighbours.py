import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#This config fixes the truncated console writes. Makes things a lot easier
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)

dataLoc = 'D:\\Course Content\\python-for-data-science-and-ml-bootcamp\\14-K-Nearest-Neighbors\\'

#Project code
#['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM', 'JHZC', 'TARGET CLASS']
my_df = pd.read_csv(dataLoc + 'KNN_Project_Data')

#Check the data for completeness (NULLS and NA's)
#sb.heatmap(data=my_df.isnull())
#plt.show()

#Any data not standardised? Standardise using a StandardScaler
my_scale = StandardScaler().fit(my_df[my_df.columns[:-1]])
my_x_data = pd.DataFrame(data=my_scale.transform(my_df[my_df.columns[:-1]]), columns=my_df.columns[:-1])

#Train/Test split of data
my_x_train, my_x_test, my_y_train, my_y_test = train_test_split(my_x_data, my_df['TARGET CLASS'])

#Find the optimal 'k' value using the elbow method
'''
error_rates = []

for i in range(1, 100):
    my_knn = KNeighborsClassifier(n_neighbors=i).fit(X=my_x_train, y=my_y_train)
    my_predictions = my_knn.predict(X=my_x_test)
    error_rates.append(np.mean(my_y_test != my_predictions))

sb.lineplot(x=range(1,100), y=error_rates, markers='o')
plt.show()
'''

#Instantiate and fit the KNN model using the k value from above
my_knn = KNeighborsClassifier(n_neighbors=20).fit(X=my_x_train, y=my_y_train)
my_predictions = my_knn.predict(X=my_x_test)

#Look at the confusion matrix and classification report, how did we do?
print(confusion_matrix(y_true=my_y_test, y_pred=my_predictions))
print('----------------------------------------------------------')
print(classification_report(y_true=my_y_test, y_pred=my_predictions))

''' ***********************************************************************
#Below is the code from the lectures

#index_col specifies to use one of the columns as the index for the dataframe
my_df = pd.read_csv(dataLoc + 'Classified Data', index_col=0)

#print(my_df.head(2))

#Scale the data to ensure that the distance function is not impacted
my_scale = StandardScaler()
my_scale.fit(my_df.drop('TARGET CLASS', axis=1)) #Exclude the dependent variable from the fit

my_scaled_df = pd.DataFrame(data=my_scale.transform(my_df.drop('TARGET CLASS', axis=1)), columns = my_df.columns[:-1])

#print(my_scaled_df.head(2))

#Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(my_scaled_df, my_df['TARGET CLASS'], test_size=0.3, random_state=101)

#Instantiate + fit model, using K = 1 (very basic)
my_knn = KNeighborsClassifier(n_neighbors=1) #n_neighbours = k
my_knn.fit(X=x_train, y=y_train)

my_predictions = my_knn.predict(X=x_test)

#print(confusion_matrix(y_true=y_test, y_pred=my_predictions))

#The elbow approach to find the optimal K value, hopefully this is the long form of it. Is very much brute force
error_rate = []

for i in range(1, 50):
    my_knn_itn = KNeighborsClassifier(n_neighbors=i).fit(X=x_train, y=y_train)
    my_knn_pred = my_knn_itn.predict(x_test)
    error_rate.append(np.mean(my_knn_pred != y_test))

sb.lineplot(x=range(1, 50), y=error_rate, markers='o')
plt.show()

'''