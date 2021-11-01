# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import joblib

from sklearn.datasets import fetch_openml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn import metrics

from sklearn.pipeline import Pipeline, make_pipeline


# openml
# https://www.openml.org/search?type=data
# this might take a few minutes
mnist = fetch_openml('Fashion-MNIST')

# what do we have
type(mnist)
dir(mnist)

# its not always garanteed, but sometimes we can just extract the data as a dataframe
fashion = mnist.frame
fashion.shape

# make it X,y
y = fashion['class']
X = fashion.drop(columns="class")

# start basic
# fit/transform or fit/predict depending on task

# not necessary, but replace all values with mean if missing
imputer = SimpleImputer()
imputer = imputer.fit(X)    # <---------- this fits the model for the features (database) - we can apply this to new, and future, datasets!
X = imputer.transform(X)    # <---------- we "fit" this model to the data, which will replace means based on the fit we learned abvoe

# above is just fine, and useful when we have train/test splits.
# we could also do fit_transform to learn and apply.  the object will still have been fit and can be reused
# X = imputer.fit_transform(X)

# scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)    # <-------- here is an example of fitting the and applying it.  scaler has learned the min/max for the features

# we could have done below before imputing/scaling, but lets break out train/test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=820, stratify=y)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# classification model, so will choose KNN classifier
clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
clf.fit(X_train, y_train)   # <------- think of this as the source of truth and comparison

# lets get predictions and probabilities  <-------- this will take some time given its a KNN model and needs to do this pairwise, even on a smaller test set
preds = clf.predict(X_test)
preds_proba = clf.predict_proba(X_test)

# how did the model do?
score = clf.score(X_test, y_test)
score

# classification report
cr = metrics.classification_report(y_test, preds)
print(cr)

# save the objects for use later
with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("knc-3.pkl", "wb") as f:
    pickle.dump(clf, f)

# WHY?  if this was the model you wanted to use in production, you can read this in when you have new data to model!


################################ Pipeline - solves the problem above

# the pipeline steps -- list of tuples with name and the bit we want to deploy, sequentially
steps = [('imputer', SimpleImputer()), 
         ('scaler', MinMaxScaler()), 
         ('clf', KNeighborsClassifier(3, metric="euclidean"))]

pipe = Pipeline(steps)

# fit the pipeline
pipe.fit(X_train, y_train)

# make the predictions
pipe_preds = pipe.predict(X_test)
pipe_probs = pipe.predict_proba(X_test)
pipe_score = pipe.score(X_test, preds)

# save the pipe
joblib.dump(pipe, "pipeline.joblib")

# ^^ Above lets us load the entire pipeline which includes data preprocessing and the model, 
#    and apply the same exactly data pipeline for inference to new data!
#    Services like Google Vertex AI API let us bring our own custom models and serve the pipeline as an API endpoint for inference
#    Just POST data of the same shape as the feature space, and scale out your API inference!
#    How cool is that?

