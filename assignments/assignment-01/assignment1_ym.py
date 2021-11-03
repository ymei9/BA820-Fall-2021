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

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn import metrics

# import os
from PIL import Image
forums = pd.read_pickle("/Users/yuxuanmei/Documents/GitHub/BA820_ym/assignments/assignment-01/forums.pkl")

# data inspection and cleaning
forums.head()
forums.info()
forums.shape
forums.describe().T

forums.set_index('text', inplace=True)
forums.isnull().sum().max()

# standardize data
scaler = StandardScaler()
scaled_forums = scaler.fit_transform(forums)

# distance
cdist = pdist(scaled_forums)

# squareform distance heatmap
sns.heatmap(squareform(cdist), cmap='Reds')
plt.show()


# hclust
hc1 = linkage(cdist)
hc1

# dendrogram
dendrogram(hc1)
plt.show()

# second hclust with distance criterion
DIST = 30
plt.figure(figsize=(8,8))
dendrogram(hc1, orientation = "left", color_threshold = DIST)

plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
plt.show()


# correlation heatmap
sns.heatmap(pd.DataFrame(scaled_forums).corr())
plt.show()

# PCA
pca = PCA(.9)
pcs = pca.fit_transform(scaled_forums)
pcs.shape

# ## what is the explianed variance ratio
# varexp = pca.explained_variance_ratio_
# varexp

 # plot
plt.title('Explained variance ration by component')
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.show()

# cumulative view
plt.title('Explained variance ration by component')
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.90)
plt.show()

# put the pcs to a new dataset
forums_pc = pd.DataFrame(pcs, index = forums.index)
forums_pc.head(3)

# tsne
tsne = TSNE()
tsne.fit(forums_pc)

# get the embeddings
te = tsne.embedding_
te.shape

# 2d tsne dataframe
tdata = pd.DataFrame(te, columns = ['e1', 'e2'])
tdata.head(3)

# tsne plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x="e1", y="e2", data=tdata, legend="full")
plt.show()

# K-means clustering
KRANGE = range(2, 15)
# containers
ss = []
avg_silo = []
for k in KRANGE: 
    km = KMeans(k)
    lab = km.fit_predict(tdata)
    avg_silo.append(metrics.silhouette_score(tdata, km.predict(tdata)))
    ss.append(km.inertia_)

sns.lineplot(KRANGE, ss)
plt.show()

sns.lineplot(KRANGE, avg_silo)
plt.show()

# choose k = 7
k7 = KMeans(7)
k7.fit(tdata)

silo_overall = metrics.silhouette_score(tdata, k7.predict(tdata))
silo_overall

silo_sample = metrics.silhouette_samples(tdata, k7.predict(tdata))
silo_sample
silo_sample.shape

skplt.metrics.plot_silhouette(tdata, k7.predict(tdata), figsize=(7,7))
plt.show()
