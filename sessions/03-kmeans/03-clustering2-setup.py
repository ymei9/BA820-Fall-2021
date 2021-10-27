# Learning goals:
## Expand on Distance and now apply Kmeans
## - Kmeans applications
## - Evaluate cluster solutions 
## - hands on with Kmeans and quick review of DBSCan

# resources
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf
# https://scikit-learn.org/stable/modules/clustering.html#dbscan


# installs
# notebook/colab
# ! pip install scikit-plot

# local/server
# pip install scikit-plot



# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt


# dataset urls:
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MedGPA.csv

judges = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820-Fall-2021/sessions/03-kmeans/bq-results-20211027-104231-rq7b7sbqro47.csv')

# lower case column names

judges.columns = judges.columns.str.lower()
judges.set_index('judge', inplace=True)
judges.head()

judges.dtypes
judges.describe().T

# fit our first kmean -- 3 clusters
k3 = KMeans(n_clusters=3, random_state=820)
KMeans()
k3.fit(judges)
k3_labs = k3.predict(judges)
k3_labs

# number of iterations took to converge
k3.n_iter_

# put clusters back into dataset
judges['k3'] = k3_labs

judges

# start to profile / learn about our cluster
judges.k3.value_counts()

judges.groupby('k3').mean().T

# fit a cluster of 5
k5=KMeans(5)
k5.fit(judges)
k5_labs = k5.predict(judges)
judges['k5'] = k5_labs
judges.groupby('k5').mean().T

# plots
k5_centers = k5.cluster_centers_
sns.scatterplot(data=judges, x="cont", y="intg", cmap="virdis", hue="k3")
plt.show()
plt.scatter(k5_centers[:,0], k5_centers[:,1], c="g", s=100)
plt.show()

sns.heatmap(judges)
plt.show()

## goodness of fit
k3.inertia_
k5.inertia_

KRANGE = range(2, 11)
# containers
ss = []
for k in KRANGE: 
    km = KMeans(k)
    lab = km.fit_predict(judges)
    ss.append(km.inertia_)
ss
sns.lineplot(KRANGE, ss)
plt.show()

silo_overall = metrics.silhouette_score(judges, k5.predict(judges))
judges.drop(columns = 'k5', inplace=True)
silo_overall

silo_sample = metrics.silhouette_samples(judges, k5.predict(judges))
silo_sample
silo_sample.shape

skplt.metrics.plot_silhouette(judges, k5.predict(judges), figsize=(7,7))
plt.show()

# useful code snippets below ---------------------------------

election = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820-Fall-2021/sessions/03-kmeans/Election08.csv')

# scale the data
el_scaler = StandardScaler()
el_scaler.fit(election)
election_scaled = el_scaler.transform(election)
