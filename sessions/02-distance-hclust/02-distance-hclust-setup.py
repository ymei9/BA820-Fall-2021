# In this session, we will explore distance calculations and their role in how we can determine 
# similarity between records.  We can use this information to segment our data into like-groups.


# resources
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html


# imports - usual suspects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for distance and h-clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

# sklearn does have some functionality too, but mostly a wrapper to scipy
from sklearn.metrics import pairwise_distances 
from sklearn.preprocessing import StandardScaler

# COLAB Only ------------------------------
# from google.colab import auth
# auth.authenticate_user()
# SQL = "SELECT * from `questrom.datasets.mtcars`"
# YOUR_BILLING_PROJECT = ""
# cars = pd.read_gbq(SQL, YOUR_BILLING_PROJECT)

cars = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820-Fall-2021/datasets/cars.csv')

# useful code snippets below ---------------------------------


# filtered cars columns
# ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear','carb']


# a plot we will see later 
# DIST = 80
# plt.figure(figsize=(5,6))
# dendrogram(hc1, 
#            labels = cars.index,
#            orientation = "left", 
#            color_threshold = DIST)
# plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
# plt.show()



# another advanced plot
# METHODS = ['single', 'complete', 'average', 'ward']
# plt.figure(figsize=(15,5))
# # loop and build our plot
# for i, m in enumerate(METHODS):
#   plt.subplot(1, 4, i+1)
#   plt.title(m)
#   dendrogram(linkage(cars_scaled.values, method=m),
#              labels = cars_scaled.index,
#              leaf_rotation=90,
#              leaf_font_size=10)
  
# plt.show()

cars.set_index('model', inplace=True)
cars.describe().T

# keep only the columns of interest
cars2  = cars[['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear','carb']]

# cars distance - passing in the numpy array from the pandas frame via values
cdist = pdist(cars2.values)

# squareform --- visualize
sns.heatmap(squareform(cdist), cmap='Reds')
plt.show()

# our first cluster!
hc1 = linkage(cdist)
hc1

# our first dendrogram!
dendrogram(hc1, labels=cars.index)
plt.show()

# a second plot

DIST = 80
plt.figure(figsize=(5,6))
dendrogram(hc1, labels = cars.index, orientation = "left", color_threshold = DIST)

plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
plt.show()

# how to identify the clusters
fcluster(hc1, 2, criterion = 'maxclust')
cars2['cluster1']  = fcluster(hc1, 2, criterion = 'maxclust')
cars2.head(3)

# 'profiling'
cars2.cluster1.value_counts()

# distance assignment
c2 = fcluster(hc1, 80, criterion = 'distance')
cars2['cluster2'] = c2
cars2.head()

# different linkage methods
linkage(cars2.values, method='')

cars2.describe()
 