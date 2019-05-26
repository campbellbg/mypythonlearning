import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#This config fixes the truncated console writes. Makes things a lot easier
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)

#!!!!!Project Exercise

file_dir = 'D:\\Course Content\\python-for-data-science-and-ml-bootcamp\\17-K-Means-Clustering\\'

#['UniName', 'Private', 'Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate', 'Room.Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate']
my_df = pd.read_csv(file_dir + 'College_Data') #This must be a built-in set of data as part of pandas

#print(my_df.columns)
#print(my_df.info())

#Rename the first column for clarity
my_df.columns = ['UniName'] + list(my_df.columns[1:])

#One-Hot encode the Private classification. Might not make any difference in this algorithm, but good practice anyway
tgt_df = pd.concat([my_df[my_df.columns[2:]], pd.get_dummies(data=my_df['Private'], drop_first=True)], axis=1)
tgt_df.columns = list(tgt_df.columns[:-1]) + ['Is_Private']

#sb.pairplot(data=tgt_df[['Is_Private', 'Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad']])
#plt.show()

#sb.scatterplot(x=tgt_df['Grad.Rate'], y=tgt_df['Room.Board'], hue=tgt_df['Is_Private'])
#plt.show()

#Just need the two clusters as the school is either private or it is not
my_model = KMeans(n_clusters=2)
my_model.fit(X=tgt_df[tgt_df.columns[:-1]])
#my_model.fit(X=tgt_df[['Grad.Rate', 'Room.Board', 'perc.alumni', 'Accept']]) #Trying a few different selective attributes to see what it does to the classification

#We can complete an evaluation as we do have some labels, which is not always the way for unsupervised data
print(np.average(my_model.labels_ == tgt_df['Is_Private']))

''' This is the work from the lecture

#Make some random data for playing the KMeans
my_data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=2, random_state=101)

#The syntax [:, 0] on a numpy array returns the first column. Not really sure what the "," is for
#plt.scatter(x=my_data[0][:, 0], y=my_data[0][:, 1], c=my_data[1])
#plt.show()

my_kmeans = KMeans(n_clusters=4)

my_kmeans.fit(my_data[0])

#print(my_kmeans.cluster_centers_) #The center of the clusters (the centroids)
#print(my_kmeans.labels_) #labels_ provides the cluster that each of the data points belongs to

print(np.average(my_data[1] == my_kmeans.labels_))

'''