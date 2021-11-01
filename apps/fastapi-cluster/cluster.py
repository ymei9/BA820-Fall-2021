"""This simple moddule will construct an artificial dataset to simulate customer personas"""

# imports
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_silhouette
import pickle

# simulate the cluster data
X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=.6, random_state=820)


# plot the data
sns.scatterplot(x=X[:, 0], y=X[:, 1]).set(title="Customer Dataset")
plt.show()

# cluster the data
# below assumes that we do not know the clusters
inertia = []
K = range(2,11)
for k in K:
    km = KMeans(k)
    labs = km.fit_predict(X)
    inertia.append(km.inertia_)


# plot this up -- interia for elbow
sns.lineplot(K, inertia)
plt.show()

# save out K5
k5 = KMeans(5)
k5.fit(X)

# plot this up -- silo score
plot_silhouette(X, k5.predict(X))
plt.show()

# save out the centers
centers = k5.cluster_centers_

# the cluster centers are what we can use to lookup new records based on euclidean distance!
with open("centers.pkl", "wb") as f:
    pickle.dump(centers, f)

