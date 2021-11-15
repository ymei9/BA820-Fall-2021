# LEARNING GOALS
#
#                 - text as a datasource
#                 - cleaning text
#                 - basic eda
#                 - Doc Term Matrix representation by hand
#                 - The intuition behind working with text before jumping into tools that abstract this away
#                 - how text can be used in ML

# some helpful resources:
# https://www.w3schools.com/python/python_regex.asp
# https://docs.python.org/3/library/re.html
# https://www.debuggex.com/cheatsheet/regex/python
# https://www.shortcutfoo.com/app/dojos/python-regex/cheatsheet

# installs
# ! pip install newspaper3k
# ! pip install -U spacy
# ! pip install wordcloud
# ! pip install emoji
# ! pip install nltk
# ! pip install scikit-plot

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

a = ['I like turtles!',
     'You like hockey and golf ',
     'Turtles and hockey ftw',
     'Python is very easy to learn. 🐍',
     'A great resource is www.spacy.io',
     ' Today is the Feb 22, 2021 !           ',
     '@username #hashtag https://www.text.com',
     'BA820 ']

df = pd.DataFrame({'text':a})
df

## QUICK QUESTION
##        What do you see about the data being brought in?

## we can always get the values back

# quick review of some of the string funcationality
# we saw in 760

# capitalize or change case
# upper, lower, strip
df.text.str.upper()
df.text.str.lower()
df.text.str.strip()


# we can detect
df.text.str.contains("hockey")

# remember python is case sensitive!

# we can replace anything that matches a pattern
# but we will come back to patterns

# we can look at the length
df.text.str.len()

#### NOTE:
##      but look at above, what do you notice about the lengths calculated?

# lets look at the values directly again for the last entry

# lets count characters and numbers
df.text.str.count("[a-zA-Z0-9]")

## regex
## https://www.regular-expressions.info/quickstart.html
##
## https://regex101.com/     <------------- fantastic resource
##
## [a-z] will match a single letter lowercase a to z
## [A-Z] will match a single letter uppercase A to Z
## [a-zA-Z0-9] will match a single character that is alphanumeric
## ^ matches a pattern at the start
## $ matches a pattern at the end
## + will match a pattern one or more times
## * will match 0 or more
## .* will match everything (dot is any character)
## {3} match pattern exactly 3 times
## {2,4} match a pattern 2 to 4 times
## {3, } match a pattern 3 or more times
## | allows us to specify "or"
## so much more including special patterns and shortcuts
## \d for a digit
## \w for word characters
## \s for whitespace

# only print out entries if the pattern matches
FIND = df.text.str.contains("tu")
df.text[FIND]

# again, case sensitive
FIND = df.text.str.contains("Tu")
df.text[FIND]

# we can use "OR" logic
FIND = df.text.str.contains("tu|BA")
df.text[FIND]

# matches

# more matches

# special characters anywhere - digits

# extract username or hashtag
# uses not whitespace character, repeating 1+



# you may get an error around capture groups
# a group is in parentheses

"""> Regular expressions and searching text can be a superpower when working with text.  If we have a large corpus, we can interate over the documents and scan/search via regular expressions to extract our datasets!"""

## Thought Exercise:
##    Our datasets that we typically see take the shape of:
##    Rows =    Observations
##    Columns = Attributes about those Observations
## 
##    How can we map this to text?
##
##    Rows =    A document (the source, we will talk about this)
##    Columns = The words in the document
##   
##    Above can be referred to as a Document Term Matrix, or Document Feature Matrix
##

# lets reset the dataframe

df = pd.DataFrame({'doc':a})
df

# if we really wanted to (or had to), we 
# have the python chops to make this a doc/term matrix

# split out the text using basic string operations
df['tokens'] = df.doc.str.split()
df.head(1)

# step 0, just the tokens but keep as a dataframe
tdf = df[['tokens']]

# step 1: melt it via explode
tdf_long = tdf.explode("tokens")
tdf_long

# step 3: back to wide for a dtm
tdf_long['value'] = 1
dtm = tdf_long.pivot_table(columns="tokens", 
                           values="value", 
                           index=tdf_long.index,
                           aggfunc=np.count_nonzero)

# lets review what we have
dtm.head(3)

## Quick thought exercise:
##      What do you notice about our tokenized dataset
##      What about the values?  What would you change?
##


dtm.fillna(0, inplace=True)
dtm.head(3)

################ YOUR TURN
##  from the topics table on big query (questrom.datasets.topics), 
##  bring in just the text column via select
##  Make the text lowercase
##  Tricky!! remove punctuation if you can (keep just letters and numbers)
##  get the text into a long form where each token is a row in the dataframe
##

SQL = "SELECT * from `questrom.datasets.topics`"
topics = pd.read_gbq(SQL, "questrom")

topics.shape
topics.sample(1)

topics['text'] = topics.text.str.lower()

# just highlighting what is possible, you don't need to do this
# keep just the numbers and letters
# just highlighting that depending on your use cases, you can 
# roll your own functions to clean text
# pandas makes it easy to `apply` these to our text column!

def remove_punct(text):
  import string
  text = ''.join([p for p in text if p not in set(string.punctuation)])
  return text

topics['text'] = topics.text.apply(remove_punct)

topics['tokens'] = topics.text.str.split()
topics.head(1)
topics_long = topics.explode("tokens")
topics_long.head(3)


#################################### Lets predict the category!
##
## we now have a dataset that can be used to fit a ML model.  
## the quality of the models and how we think about ML tasks is all about the data
## let's start with this framing for intuition
##
##

## get the topics data again

# topics = pd.read_gbq("SELECT * from `questrom.datasets.topics`", "questrom")
# topics.shape

del topics['tokens']

# what do we have

# what do we have for a distro on topics?
topics.topic.value_counts(dropna=False)

# imports -- violating my rule of thumb, but lets put that aside for emphasis

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# remember, we have the topics data

# we can tokenize our data with sklearn pipelines
# above highlights we have full control, but there are frameworks that aim to abstract this for us
# abstractions have their own overhead costs, but lets build on top of sklearn to soften the impact

cv = CountVectorizer()
cv.fit(topics.text)

# we can easily have done fit_transform, but lets explore what was learned about our corpus

# get the vocabulary and their term:numeric id map
# this is a common representation for downstream word embedding tasks
cv.vocabulary_

# length
len(cv.vocabulary_)

## make this a numeric matrix of document by term (dtm)
dtm = cv.transform(topics.text)

# confirm the shape is what we expect
dtm.shape
type(dtm)

# missing data are zeros
dtm.toarray()[:5, :5]

# make this a dataframe to help with our mental model

dtm_df = pd.DataFrame(dtm.toarray(), columns=cv.get_feature_names())
dtm_df.columns

# lets build the datasets for the model

X = dtm_df.copy()
y = topics.topic

# confirm we have the same thing
X.shape
y.shape

# split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=820, stratify=y)

# fit the model

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=30, min_samples_leaf=15)
tree.fit(X_train, y_train)

# fit metrics on test

preds = tree.predict(X_test)
ctable = metrics.classification_report(y_test, preds)
print(ctable)

# confusion matrix from skplot
# cancan see where the model isn't sure

skplot.metrics.plot_confusion_matrix(y_test, preds, 
                                     figsize=(7,4), 
                                     x_tick_rotation=90 )
plt.show()

# accuracy score   <----- confirming the classification report

#################################### REVIEW
##
## - normal text form -> a DTM
## - we saw that tokenizing, and the logic we apply, matters (case, punctuation)
#     will we see even more example
## - if we had to, we can parse text into a format for machine learning
## - nothing stopping us from passing in a count-based dtm into a ML model!
##

############################################################
########################################### Team Challenge
############################################################
# 
## Work in Project Groups
# 
# - tokenize the dataset on Big Query from 
# URL link: https://console.cloud.google.com/bigquery?project=questrom&d=SMSspam&

## review the slides at the end of this module
## predict spam
## objetive =  based on f1
## only input is text, but you can derive features
## limited time, but how do you maximize your time (and the model?)
## HINTS:
##        start small, simple models
##        iterate and see how you do against the leaderboard
##        code above helps you with the core mechanics

# get the datasets - select * is fine, but there are two datasets and an example submission to review!





"""! head myteam-submission.csv"""

