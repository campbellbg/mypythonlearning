import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.datasets import load_breast_cancer #built in datasets that come with skikit learn (sklearn)

#This config fixes the truncated console writes. Makes things a lot easier
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)

#['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
my_data = load_breast_cancer() #datatype is a little odd. Works like a dictionary, however, which can then be used to get the data out that we need

my_df_var = pd.DataFrame(data=my_data['data'], columns=my_data['feature_names'])

#!!!!!!! Scale the data !!!!!!!!!!!!

#Standard scalar fitted to ensure that all data in scaled in the same manner
my_scalar = StandardScaler().fit(my_df_var)

my_df_scaled = pd.DataFrame(data=my_scalar.transform(my_df_var), columns=my_data['feature_names']) #Scaled data

#!!!!! Perform the PCA !!!!!!!!!!!!
my_pca = PCA(n_components=2).fit(my_df_scaled) #Maintain the two first PC's

my_df_pca = my_pca.transform(my_df_scaled)

#plt.scatter(x=my_df_pca[:, 0], y=my_df_pca[:, 1], c=my_data['target']) #I am able to use a completely different dataset for colour as it joins on index, which is maintained
#plt.show()

print(my_pca.components_)



