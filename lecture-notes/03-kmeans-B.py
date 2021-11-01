# import numpy as np
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

# pip install scikit-plot
# ! pip install scikit-plot





# WARMUP EXERICSE:
# dataset:
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv

# pip install scikit-plot
# ! pip install scikit-plot

# task
# use hierarchical clustering on the election dataset
# keep just the numerical columns
# add state abbreviation as the index
# use complete linkage and generate 4 clusters
# put back onto the original dataset
# profile the number of states by cluster assignment and the % that Obama won


# read in the dataset
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv"
eo = pd.read_csv(URL)
eo.shape
eo.sample(3)

# data cleaning
# clean the column names
# change the index
# 

eo.columns = eo.columns.str.lower()

eo.index = eo.abr

election = eo.loc[:, "income":"dem.rep"]

# standardized 
scaler = StandardScaler()
escaled = scaler.fit_transform(election)

# cluster now
hc1 = linkage(escaled, method="complete")
hc1

# plot 
plt.figure(figsize=(15, 5))
dendrogram(hc1, labels=election.index)
plt.show()

# extract 4 clusters
eo['cluster'] = fcluster(hc1, 4, criterion="maxclust")
eo.sample(3)

# value counts
eo.cluster.value_counts()

# profiling
eo.groupby("cluster")['obamawin'].mean()
eo.groupby("cluster").describe()



#### kmeans

PROJECT = 'questrom'    # <------ change to your project
SQL = "SELECT * from `questrom.datasets.judges`"
judges = pd.read_gbq(SQL, PROJECT)

judges.shape
judges.sample(3)

# data cleanup
# clean up column colnames
# judge index -- numeric
judges.index = judges.judge
del judges['judge']
judges.columns = judges.columns.str.lower()

# quick look at the data
judges.describe().T

# fit our first kmeans cluster!
k3 = KMeans(3)
k3.fit(judges)
labs = k3.predict(judges)
labs

# how many iterations ran
k3.n_iter_

# put these back onto the original dataset
judges['k3'] = labs
judges.sample(3)

# lets do our first profile
k3_profile = judges.groupby("k3").mean()
k3_profile.T

sns.heatmap(k3_profile, cmap="Reds")
plt.show()

# typing tools
judges.sample(3)

## your turn
## fit a cluster solution with 5 clusters
## apply it back it to the dataset
## do a quick profile/persona of the clusters

k5 = KMeans(5)
j = judges.copy()
del j['k3']

judges['k5'] = k5.fit_predict(j)

# inertia value for k5
k5.inertia_
k3.inertia_ 


## exercise
## fit range of cluster solutions for 2 to 10, k=2, k=3
## save out a way to evaluate the solutions based on the interia of the fit
## 

KRANGE = range(2, 11)

# a container
ss = []

for k in KRANGE:
    km = KMeans(k)
    lab = km.fit_predict(j)
    ss.append(km.inertia_)

sns.lineplot(KRANGE, ss)
plt.show()


# silo score (fit and the samples) comes from the metrics module
k5.inertia_

silo_overall = metrics.silhouette_score(j, k5.predict(j))
silo_overall

# samples
silo_samples = metrics.silhouette_samples(j, k5.predict(j))
silo_samples.shape

# plotting
skplt.metrics.plot_silhouette(j, k5.predict(j), figsize=(7,7))
plt.show()

