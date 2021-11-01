######################################################
## Wrap up UML
## Hands on-heavy class to dive into some concepts you may want to explore 
## Learning objectives:
##
## 0. wrap up PCA
## 1. exposure to more contemporary techniques for interviews, awareness, and further exploration
## 2. highlight different use-cases, some that help with viz for non-tech, others to think about alternatives to linear PCA
## 3. use this as a jumping off point for you to consider the fact that are lots of methods, and its not typically for this task, do this one approach
######################################################

## resources
# - https://pair-code.github.io/understanding-umap/
# - https://distill.pub/2016/misread-tsne/
# - repo has a resources folder, review the about.md file for additional links.


# installs
# ! pip install umap-learn
# pip install umap-learn

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# scipy
from scipy.spatial.distance import pdist

# scikit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.cluster import KMeans



##############################################################################
## Code samples
##############################################################################


############################################ mnist data
# from sklearn.datasets import load_digits
# digits = load_digits()
# X = digits.data
# y = digits.target
# img = digits.images[0]
# img.shape
# plt.imshow(img, cmap="gray")
# plt.title(f"Label: {y[0]}")
# plt.show()
# img.flatten()



############################################ decision tree
# from sklearn.tree import DecisionTreeClassifier  # simple decision tree
# tree = DecisionTreeClassifier(max_depth=5)   # max depth of tree of 4 is random for example
# tree.fit(X, y)  # sklearn syntax is everywhere!
# tree_preds = tree.predict(X)   # 
# tree_acc = tree.score(X, y)
# tree_acc




############################################ tsne
# tsne = TSNE()
# tsne.fit(pcs_m)
# te = tsne.embedding_

# tdata = pd.DataFrame(te, columns=["e1", "e2"])
# tdata['y'] = y


############################################ seaborn mnist plot
# PAL = sns.color_palette("bright", 10) 
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x="e1", y="e2", hue="y", data=tdata, legend="full", palette=PAL)


############################################ umap
# from umap import UMAP
# u = UMAP(random_state=820, n_neighbors=10)
# u.fit(X)
# embeds = u.transform(X)




##############################################################################
## Other Considerations for UML
##############################################################################
##
##  Other iterations of PCA even
##     - Randomized PCA (generalizes and approximates for larger datasets)
##     - Incremental PCA (helps when the data can't fit in memory)
##
##  Recommendation Engines
##      - extend "offline" association rules 
##      - added some links to the resources (great article with other libraries)
##      - toolkits exist to configure good approaches for real-use
##      - I call reco engines unsupervised because its moreso about using neighbors and similarity to back 
##        into items to recommend
##      - can be done by finding similar users, or similar items.
##      - hybrid approaches work too
##      - scikit surprise
##      NOTE:  Think about it? you can pull data from databases!  Build your own reco tool by running a simple API!
##             batch calculate recos and store in a table, send user id to API, look up the previously made recommendations
##             post feedback to database, evaluate, iterate, repeat!
##     
##   A python package to review
##      - I followed this package in its early days (graphlab) before Apple bought the company
##      - expressive, with pandas-like syntax
##      - accessible toolkit for a number of ML tasks, including Reco engines 
##      - https://github.com/apple/turicreate



################### Breakout Challenge
## work as a group to combine UML and SML!
## housing-prices tables on Big Query will be used questrom.datasets._______
##     housing-train = your training set
##     housing-test = the test set, does not have the target, BUT does have an ID that you will need for your submission
##     housing-sample-submission = a sample of what a submission file should look like, note the id and your predicted value
## 
## use regression to predict median_house_value
## 
## you can use all of the techniques covered in the program, and this course
## objective:  MAE - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
##
##
## tips/tricks
##    - ITERATE!  iteration is natural, and your friend
##    - submit multiple times with the same team name
##    - what would you guess without a model, start there!
##    - you will need to submit for all IDs in the test file
##    - it will error on submission if you don't submit 
##
## Leaderboard and submissions here: TBD
## 