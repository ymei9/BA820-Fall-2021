# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot

# some "fun" new packages
from wordcloud import WordCloud
import emoji

import re

# new imports for text specific tasks
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk

train = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820_ym/data-challenges/train.csv')
test = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820_ym/data-challenges/test.csv')

train.isnull().sum()
train.info()

# preprecossing 
train['message'] = train.message.str.lower()

# def remove_punct(text):
#     import string
#     text = ''.join([p for p in text if p not in set(string.punctuation)])
#     return text
# train['message'] = train.message.apply(remove_punct)

# # tokenize
# train['tokens'] = train.message.str.split()
# t_train = train[['tokens']]
# train_long = t_train.explode("tokens")

# train_long['value'] = 1
# train_long
# dtm = train_long.pivot_table(columns="tokens", 
#                            values="value", 
#                            index=train_long.index,
#                            aggfunc=np.count_nonzero)
# dtm.info()
# dtm.fillna(0, inplace=True)
# dtm.head(3)




## use CountVectorizer 


cv = CountVectorizer()
cv.fit(train.message)

cv.vocabulary_
# length
len(cv.vocabulary_)

## make this a numeric matrix of document by term (dtm)
dtm_cv = cv.transform(train.message)

# confirm the shape is what we expect
dtm_cv.shape
train.shape

# missing data are zeros
dtm.toarray()[:5,:5]
type(dtm)

# make this a dataframe to help with our mental model

dtm_df = pd.DataFrame(dtm_cv.toarray(), columns=cv.get_feature_names())
dtm_df.columns


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics



# split train test data
X = dtm_df.copy()
y = train.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


rf = RandomForestClassifier()
from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
    n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)
rf_random.best_params_

rf_hp = RandomForestClassifier(n_estimators = 1000, min_samples_split = 10, min_samples_leaf = 1, 
        max_features ='sqrt', max_depth = 80, bootstrap = False)
rf_hp.fit(X_train, y_train)
preds = rf_hp.predict(X_test)
ctable = metrics.classification_report(y_test, preds)
print(ctable)

# confusion matrix from skplot
# cancan see where the model isn't sure

skplot.metrics.plot_confusion_matrix(y_test, preds, 
                                     figsize=(7,4), 
                                     x_tick_rotation=90 )
plt.show()

# accuracy score   <----- confirming the classification report
rf_hp.score(X_test, y_test)

## make this a numeric matrix of document by term (dtm)
dtm_test = cv.transform(test.message)

# make this a dataframe to help with our mental model

dtm_test = pd.DataFrame(dtm_test.toarray(), columns=cv.get_feature_names())

preds_test = rf_hp.predict(dtm_test)
preds_test
test['label'] = preds_test
J = test.set_index('id')
J.drop(columns = 'message', inplace=True)
J.to_csv('J.csv')
J