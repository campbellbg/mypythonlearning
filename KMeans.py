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