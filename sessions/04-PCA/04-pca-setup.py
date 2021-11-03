##############################################################################
## Dimension Reduction 1: Principal Components Analysis
## Learning goals:
## - application of PCA in python via sklearn
## - data considerations and assessment of fit
## - hands on data challenge = Put all of your skills from all courses together!
## - setup non-linear discussion for next session
##
##############################################################################

# installs

# notebook/colab
# ! pip install scikit-plot
# pip install scikit-plot

# imports
import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import Scalar
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
# 4. generate the silohouette plot for the solution
# 5. create a boxplot of the column carat by cluster label (one boxplot for each cluster)

diamonds = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820-Fall-2021/sessions/04-PCA/diamonds.csv')
diamonds = diamonds.select_dtypes(['number'])

# standardize the data
sclaer = StandardScaler()
sclaer.fit(diamonds)
dia_scaled = sclaer.transform(diamonds)

dia_scaled

# fit a cluster of 5
k5 = KMeans(n_clusters=5)
KMeans()
k5.fit(dia_scaled)
k5_labs = k5.predict(dia_scaled)
k5_labs

# append on original dataset
diamonds['k5'] = k5_labs
diamonds

# boxplot against carat
import seaborn as sns
sns.boxplot(data = diamonds, x ='k5', y = 'carat')
plt.show()

# silohouette plot
skplt.metrics.plot_silhouette(dia_scaled, k5_labs, figsize=(7,7))
plt.show()

k5.inertia_

##############################################################################
## Code snippets for our discussion
##############################################################################

################# quick function to construct the barplot easily
# def ev_plot(ev):
#   y = list(ev)
#   x = list(range(1,len(ev)+1))
#   return x, y

# x, y = ev_plot(pca.explained_variance_)

# plt.title("Explained Variance - Eigenvalue")
# plt.bar(x=x, height=y)
# plt.axhline(y=1, ls="--")


################# loadings matrix
# component, feature
# comps = pca.components_
# COLS = ["PC" + str(i) for i in range(1, len(comps)+1)]
# loadings = pd.DataFrame(comps.T, columns=COLS, index=judges.columns)


################# categorical data for diamonds dataset plot of PC2
# dia['cut2'] = dia.cut.astype('category').cat.codes
# plt.scatter(x=dia.pc2_1, y=dia.pc2_2, c=dia.cut2, cmap=cm.Paired, alpha=.3)






##############################################################################
## PRACTICE: Data Exercise
##############################################################################

## - Diamonds data challenge in breakout rooms
## - lets start to see how we can combine UML and SML!
##
## - OBJECTIVE:  As a group, fit a regression model to the price column
## -             What is your R2? can you beat your best score?
## 
##
## 1. refit PCA to the diamonds dataset.
## 2. how many components would you select
## 3. remember!  do not include price in your columns when generating the components, we are predicint that!
## 4. Iterate!  try different models, assumptions
##
## NOTE:  we haven't covered regression in scikit learn, but its the same flow!
## Some help:  
##   
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score 

# in this case, does dimensionality reduction help us?

judges = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820-Fall-2021/sessions/03-kmeans/bq-results-20211027-104231-rq7b7sbqro47.csv')

judges.index = judges.judge
del judges['judge']
judges.info()
judges.head()
judges.describe().T

# correlation matrix
jc = judges.corr()
sns.heatmap(jc, cmap='Reds', center=0)
plt.show()

# fit our first model for PCA
 pca = PCA()
 pcs = pca.fit_transform(judges)
 pcs.shape

 ## what is the explianed variance ratio
 varexp = pca.explained_variance_ratio_
 varexp

 # plot
 plt.title('Explained variance ration by component')
 sns.lineplot(range(1, len(varexp)+1), varexp)
 plt.show()

# cumulative view
plt.title('Explained variance ration by component')
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95)
plt.show()

# explained variance (not ratio)
# use a threshold of 1, not ratio
expvar = pca.explained_variance_

plt.title('Explained variance by component')
sns.lineplot(range(1, len(expvar)+1), expvar)
plt.axhline(1)
plt.show()

# diamond dataset
d2 = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820-Fall-2021/sessions/04-PCA/diamonds.csv')
d2 = d2.select_dtypes(['number'])
d2

# rescale
sclaer = StandardScaler()
sclaer.fit(d2)
ds2 = sclaer.transform(d2)
df = pd.DataFrame(data = ds2)
df.describe().T

pca = PCA()
pc_d2 = pca.fit_transform(ds2)
pc_d2.shape

## what is the explianed variance ratio
varexp_d2 = pca.explained_variance_ratio_
varexp_d2

# plot
plt.title('Explained variance ration by component')
sns.lineplot(range(1, len(varexp_d2)+1), varexp_d2)
plt.show()

# cumulative view
plt.title('Explained variance ration by component')
sns.lineplot(range(1, len(varexp_d2)+1), np.cumsum(varexp_d2))
plt.axhline(.95)
plt.show()

# explained variance (not ratio)
# use a eigenvalue threshold of 1, not ratio
expvar_d2 = pca.explained_variance_

plt.title('Explained variance by component')
sns.lineplot(range(1, len(expvar_d2)+1), expvar_d2)
plt.axhline(1)
plt.show()

###### judges
pca.n_components_
comps = pca.components_
COLS = ['PC' + str(i) for i in range(1, len(comps)+1)]

loadings = pd.DataFrame(comps.T, columns = COLS, index=judges.T.index)
loadings

# plot 
sns.heatmap(loadings, cmap = 'vlag')
plt.show()

# put the pcs to a new dataset
comps_judges = pcs[:,:2]
j = pd.DataFrame(comps_judges, columns = ['c1','c2'], index = judges.index)

sns.scatterplot(data=j, x='c1',y='c2')
plt.show()