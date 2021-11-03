import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt

forums = pd.read_pickle("/Users/yuxuanmei/Documents/GitHub/BA820-Fall-2021/assignments/assignment-01/forums.pkl")

# data inspection and cleaning
forums.head()
forums.info()
forums.describe().T

forums.set_index('text', inplace=True)
forums.isnull().sum().max()

# standardize data
scaler = StandardScaler()
scaled_forums = scaler.fit_transform(forums)

# distance
cdist = pdist(scaled_forums)

# squareform --- visualize
sns.heatmap(squareform(cdist), cmap='Reds')
plt.show()


# hclust
hc1 = linkage(cdist)
hc1

# our first dendrogram!
dendrogram(hc1, labels= range(0, 2362))
plt.show()

# a second plot

DIST = 80
plt.figure(figsize=(5,6))
dendrogram(hc1, labels = forums.index, orientation = "left", color_threshold = DIST)

plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
plt.show()