import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

# data ingestion
forums = pd.read_pickle("/Users/yuxuanmei/Documents/GitHub/BA820_ym/assignments/assignment-01/forums.pkl")

# data inspection and cleaning
forums.head()
forums.info()
forums.shape
forums.describe().T

# drop duplicated rows
forums[forums.duplicated()]
forums.drop_duplicates(inplace=True)
forums.set_index('text', inplace=True)
forums.isnull().sum().max()
forums

# standardize data
scaler = StandardScaler()
scaled_forums = scaler.fit_transform(forums)
scaled_forums.shape

# distance
cdist = pdist(scaled_forums)

# squareform distance heatmap
sns.heatmap(squareform(cdist), cmap='Reds')
plt.show()


# hclust
hc1 = linkage(cdist, method='ward')
hc1

# dendrogram
plt.title('Hierarchical clustering on scaled data')
dendrogram(hc1)
plt.show()

# second hclust with distance criterion as 150
DIST = 150

plt.figure(figsize=(8,8))
plt.title('Hierarchical clustering on scaled data with distance criterion')
dendrogram(hc1, color_threshold = DIST)

plt.axhline(y=DIST, c='red', lw=2, linestyle='dashed')
plt.show()


# correlation heatmap
sns.heatmap(pd.DataFrame(scaled_forums).corr())
plt.show()

# K-means cluster without PCA

KRANGE = range(2, 15)
# containers for inertia and silhouette scores
ss1 = []
avg_silo1 = []
for k in KRANGE: 
    km = KMeans(k)
    lab = km.fit_predict(scaled_forums)
    avg_silo1.append(metrics.silhouette_score(scaled_forums, km.predict(scaled_forums)))
    ss1.append(km.inertia_)

plt.title('Inertia on sclaed dataset')
sns.lineplot(KRANGE, ss1)
plt.show()

plt.title('Silhouette score on scaled dataset')
sns.lineplot(KRANGE, avg_silo1)
plt.show()
# clustering returned extremly poor result before running PCA and tsne

############

# PCA
pca = PCA(70)
pcs = pca.fit_transform(scaled_forums)
pcs.shape

## what is the explianed variance ratio
varexp = pca.explained_variance_ratio_
varexp

 # plot
plt.title('Explained variance ratio by component')
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.show()

# cumulative view
plt.title('Cumulative explained variance ration by component')
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.85, color = 'red', linestyle = 'dashed')
plt.show()

# put the pcs to a new dataset
forums_pc = pd.DataFrame(pcs, index = forums.index)
forums_pc.head(3)

#########

# tsne analysis to further reduce demension
tsne = TSNE(perplexity=50)
tsne.fit(forums_pc)

# get the embeddings
te = tsne.embedding_
te.shape

# 2d tsne dataframe
tdata = pd.DataFrame(te, columns = ['e1', 'e2'])
tdata.head(3)

# tsne plot
plt.figure(figsize=(10, 8))
plt.title('TSNE plot')
sns.scatterplot(x="e1", y="e2", data=tdata, legend="full")
plt.show()

# K-means clustering
KRANGE = range(2, 15)
# containers for inertia and silhouette scores
ss_tsne = []
avg_silo_tsne = []
for k in KRANGE: 
    km = KMeans(k)
    lab = km.fit_predict(tdata)
    avg_silo_tsne.append(metrics.silhouette_score(tdata, km.predict(tdata)))
    ss_tsne.append(km.inertia_)

plt.title('Inertia with PCA and TSNE')
sns.lineplot(KRANGE, ss_tsne)
plt.axvline(x = 6, linestyle = 'dashed', color = 'red')
plt.axvline(x = 5, linestyle = 'dashed', color = 'green')
plt.show()

plt.title('Silhouette score with PCA and TSNE')
sns.lineplot(KRANGE, avg_silo_tsne)
plt.axvline(x = 6, linestyle = 'dashed', color = 'red')
plt.axvline(x = 5, linestyle = 'dashed', color = 'green')
plt.show()

# choose k = 5
k5 = KMeans(5)
k5.fit(tdata)

# check the silhouette score
silo_overall = metrics.silhouette_score(tdata, k5.predict(tdata))
silo_overall

silo_sample = metrics.silhouette_samples(tdata, k5.predict(tdata))
silo_sample
silo_sample.shape

skplt.metrics.plot_silhouette(tdata, k5.predict(tdata), figsize=(7,7))
plt.show()

# cluster with k = 5
lab_final = k5.predict(tdata)
tdata['label'] = lab_final

plt.title('Cluster result')
sns.scatterplot(data=tdata, x='e1', y='e2', hue='label')
plt.show()


#####################
#####################

# do everything again on unscaled data (testing purpose)
# # PCA
# pca2 = PCA(.9)
# pcs2 = pca2.fit_transform(forums)
# forums
# pcs2.shape

# # put the pcs to a new dataset
# forums_pc2 = pd.DataFrame(pcs2, index = forums.index)
# forums_pc2.head(3)

# TSNE()

# # tsne analysis to further reduce demension
# tsne2 = TSNE(perplexity=50)
# tsne2.fit(forums_pc2)

# # get the embeddings
# te2 = tsne2.embedding_
# te2.shape

# # 2d tsne dataframe
# tdata2 = pd.DataFrame(te2, columns = ['e1', 'e2'])
# tdata2.head(3)

# # # tsne plot
# # plt.figure(figsize=(10, 8))
# # plt.title('TSNE plot')
# # sns.scatterplot(x="e1", y="e2", data=tdata, legend="full")
# # plt.show()

# # K-means clustering
# KRANGE = range(2, 15)
# # containers for inertia and silhouette scores
# ss_tsne2 = []
# avg_silo_tsne2 = []
# for k in KRANGE: 
#     km = KMeans(k)
#     lab = km.fit_predict(tdata2)
#     avg_silo_tsne2.append(metrics.silhouette_score(tdata2, km.predict(tdata2)))
#     ss_tsne2.append(km.inertia_)

# plt.title('Inertia with PCA and TSNE')
# sns.lineplot(KRANGE, ss_tsne2)
# plt.axvline(x = 6, linestyle = 'dashed', color = 'red')
# plt.axvline(x = 7, linestyle = 'dashed', color = 'green')
# plt.show()

# plt.title('Silhouette score with PCA and TSNE')
# sns.lineplot(KRANGE, avg_silo_tsne2)
# plt.axvline(x = 6, linestyle = 'dashed', color = 'red')
# plt.axvline(x = 7, linestyle = 'dashed', color = 'green')
# plt.show()

# # choose k = 7
# k7 = KMeans(7)
# k7.fit(tdata2)

# # check the silhouette score
# silo_overall = metrics.silhouette_score(tdata, k6.predict(tdata))
# silo_overall

# silo_sample = metrics.silhouette_samples(tdata, k6.predict(tdata))
# silo_sample
# silo_sample.shape

# skplt.metrics.plot_silhouette(tdata2, k7.predict(tdata2), figsize=(7,7))
# plt.show()