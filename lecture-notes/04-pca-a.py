##############################################################################
## Dimension Reduction 1: Principal Components Analysis
## Learning goals:
## - application of PCA in python via sklearn
## - data considerations and assessment of fit
## - hands on data challenge = Put all of your skills from all courses together!
## - setup non-linear discussion for next session
##
##############################################################################



# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics 

import scikitplot as skplt

# color maps
from matplotlib import cm


# resources
# Seaborn color maps/palettes:  https://seaborn.pydata.org/tutorial/color_palettes.html
# Matplotlib color maps:  https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Good discussion on loadings: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html


##############################################################################
## Warmup Exercise
##############################################################################

# warmup exercise
# questrom.datasets.diamonds
# 1. write SQL to get the diamonds table from Big Query
# 2. keep only numeric columns (pandas can be your friend here!)
# 3. use kmeans to fit a 5 cluster solution
# 4. create a boxplot of the column carat by cluster label (one boxplot for each cluster)
# Optional. generate the silohouette plot for the solution (might take a few minutes)

SQL = "SELECT * from `questrom.datasets.diamonds`"
dia = pd.read_gbq(SQL, "questrom")
dia.info()

# keep numeric variables
dia_num = dia.select_dtypes("number")

# standardize the data
scaler = StandardScaler()
scaler.fit(dia_num)
dia_scaled = scaler.transform(dia_num)

# dia_scaled = scaler.fit_transform(dia_num)

k5 = KMeans(5)
k5_labs = k5.fit_predict(dia_scaled)
np.unique(k5_labs)

# append to the original dataset
dia['k5'] = k5_labs

# boxplot against the variable carat
sns.boxplot(data=dia, x="k5", y="carat")
plt.show()

# OPTIONAL:
# skplot.metrics.plot_silhouette(dia_scaled, k5_labs)

k5.inertia_



################################ PCA

SQL = "SELECT * from `questrom.datasets.judges`"
judges = pd.read_gbq(SQL, "questrom")

judges.info()

judges.index = judges.judge
del judges['judge']
judges.info()
judges.head(3)

judges.describe().T

# correlation matrix
jc = judges.corr()
sns.heatmap(jc, cmap="Reds", center=0)
plt.show()
# plt.clf()

# fit our first model for PCA
pca = PCA()
pcs = pca.fit_transform(judges)

pcs.shape
type(pcs)


## what is the explained variance ratio
varexp = pca.explained_variance_ratio_
type(varexp)
varexp.shape


# plot 
plt.title("Explained Variance Ratio by Component")
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.show()

# cumulative view
plt.title("Explained Variance Ratio by Component")
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95)
plt.show()

# explained variance (not ratio)
explvar =  pca.explained_variance_
type(explvar)
explvar.shape

plt.title("Explained Variance Ratio by Component")
sns.lineplot(range(1, len(varexp)+1), explvar)
plt.axhline(1)
plt.show()


## PCA on the diamonds the dataset
## grab the diamonds dataset from big query
## do any data preprocessing?
## fit a PCA a model to the data
## how many PCs would extract

# dia
dia.shape
dia.head(3)

del dia['k5']

dia.info()

dia_num.info()

# scale
pca_dia = PCA()
pcs_dia = pca_dia.fit_transform(dia_scaled)

pcs_dia.shape

dia_expvar = pca_dia.explained_variance_ratio_
plt.title("Explained Variance Ratio by Component")
sns.lineplot(range(1, len(dia_expvar)+1), dia_expvar)
plt.show()

dia_eval = pca_dia.explained_variance_
sns.lineplot(range(1, len(dia_eval)+1), dia_eval)
plt.axhline(1)
plt.show()


########## judges

pca.n_components_

comps = pca.components_

COLS = ["PC" + str(i) for i in range(1, len(comps)+1)]

loadings = pd.DataFrame(comps.T, columns=COLS, index=judges.columns)
loadings

# plot of this
sns.heatmap(loadings, cmap="vlag")
plt.show()

# matches the shape of the judges
pcs.shape
judges.shape

# put this back onto a new dataset
comps_judges = pcs[:, :2]
comps_judges.shape

j = pd.DataFrame(comps_judges, columns = ['c1', 'c2'], index=judges.index)
j.head(3)

sns.scatterplot(data=j, x="c1", y="c2")
plt.show()


