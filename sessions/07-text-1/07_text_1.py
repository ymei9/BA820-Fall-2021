# LEARNING GOALS
#
#                 - text as a datasource
#                 - cleaning text
#                 - basic eda
#                 - Doc Term Matrix representation by hand
#                 - The intuition behind working with text before jumping into tools that abstract this away
#                 - how text can be used in ML

######################################## some helpful resources:
# https://www.w3schools.com/python/python_regex.asp
# https://docs.python.org/3/library/re.html
# https://www.debuggex.com/cheatsheet/regex/python
# https://www.shortcutfoo.com/app/dojos/python-regex/cheatsheet

# installs
# ! pip install newspaper3k
# ! pip install spacy
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
from newspaper import Article

# some "fun" packages
from wordcloud import WordCloud
import emoji

import re

# new imports
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk

# a simple corpus
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

# capitalize or change case
# upper, lower, strip

# we can detect

# remember python is case sensitive!

# we can replace anything that matches a pattern
# but we will come back to patterns

# we can look at the length

#### NOTE:
##      but look at above, what do you notice about the lengths calculated?

# lets look at the values directly again for the last entry

# lets count characters and numbers

# df.text.str.count("[a-zA-Z0-9]")

## regex
## https://www.regular-expressions.info/quickstart.html
##
## https://regex101.com/
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
## [:punct:]

# only print out entries if the pattern matches

# again, case sensitive

# we can use "OR" logic

# FIND = df.text.str.contains("tu+|BA")
# df.text[FIND]

# matches

# FIND = df.text.str.contains("hock{1}")
# df.text[FIND]

# matches

# FIND = df.text.str.contains("\d")
# df.text[FIND]

# special characters anywhere
# FIND = df.text.str.contains("\d")
# df.text[FIND]

# extract username or hashtag
# uses not whitespace character, repeating 1+
# df.text.str.findall('@\S+|#\S+')



# you may get an error around capture groups
# a group is in parentheses

# df.text.str.extract('@\S+')    #<----- error
# df.text.str.extract('(@\S+)')   # <- works as expected

########################################### Regex Exercise
##
##  ~ 15 minutes to get back into the flow of text and the skills we saw earlier
##
##  There is a story (article) at the URL below
##  Let's go back to some of the material that we covered in BA765
##  Pull in the article into a format that you can analyze the full story
##  Below are various questions for you to consider as your parse the information
##  This helps us think about the elements that exist in text, and how 
##  we can consider extracting features from text

## HINT/TIP!  We used newspaper and requests/Beautifulsoup last semester

URL = "https://news.yahoo.com/world-first-airport-therapy-pig-131238990.html"

# get the article and parse
# article = Article(URL)
# article.download()
# article.parse()

# the text into a pandas dataframe
# txt = [article.text]
# txt = pd.DataFrame({"txt":txt})

# what do we have

################################################
## ok, lets start to think of text as a dataset
## we will smart small, and then start to increment
##

docs = ['I like golf', 
        'i like hockey',
        'Data science is a super skill!']

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

# remember split?





# lets do this in a dataframe

# df = pd.DataFrame({'doc':docs})
# df
# df['tokens'] = df.doc.str.split()

# if we really wanted to (or had to), we 
# have the python chops to make this a doc/term matrix

# step 0, just the tokens but keep as a dataframe
# tdf = df[['tokens']]

# step 1: melt it via explode
# tdf_long = tdf.explode("tokens")
# tdf_long

# step 3: back to wide for a dtm

# tdf_long['value'] = 1
# dtm = tdf_long.pivot_table(columns="tokens", 
#                            values="value", 
#                            index=tdf_long.index,
#                            aggfunc=np.count_nonzero)



# dtm

# lets review what we have

## Quick thought exercise:
##      What do you notice about our tokenized dataset
##      What about the values?  What would you change?
##

################ YOUR TURN
##  from the topics table on big query (datasets.topics), bring in just the text column
##  Make the text lowercase
##  Tricky!! remove punctuation if you can (keep just letters and numbers)
##  get the text into a long form where each token is a row in the dataframe
##

################################### Text EDA
## lets get a cleaned dataset in long format

# get the data from topics_long

# remember!  use your billing project

# what do we have

### ## in case, this is what some in the R space might call a tidy dataset

# lets just recenter what we have

# start simple, how many unique tokens

# what are the top 15 words

# we could even plot this if we had to

# lets plot all word frequencies

# (tl
#  .token
#  .value_counts()
#  .plot(kind="line"))

## Plot is a little misleading, but the idea is the most
## common words occur at rates much larger than other words
##
## This is where the typical discussion of Zipfs Law is introduced
##  Frequency of a word is inversely proportional to its rank
## Most common word appears approx. twice as often as the 2nd most common word, etc.
## Most texts stop at this point, but I linked to a good article on Medium on the "why"
## TLDR: - we can make inferences about the probability of seeing a given word with a corpus of N size of words
##       - it matters because when we define our vocab size, this can help us think about sample sizes for our model
##       - this is where big data/deep learning comes into play, but we can go A LONG way prior to getting there

# quick review: what does our data look like again

# summary stats

# N_TOPICS = tl.topic.nunique()
# N_DOCS = tl.id.nunique()
# print(f"number of unique topics: {N_TOPICS}")
# print(f"number of documents in the corpora: {N_DOCS}")

# we could even summarize stats about each document
# token_stats = tl.groupby("token", as_index=False)
# token_stats = token_stats.agg({'id':['size', 'nunique'],'topic':'nunique'})

# clean cols to start
# new_cols = ['_'.join(t).rstrip("_") for t in token_stats.columns]
# token_stats.columns = new_cols

# new summary cols
# token_stats['pct_docs'] = token_stats.id_nunique / N_DOCS  ## token occurrence % all docs
# token_stats['pct_topics'] = token_stats.topic_nunique / N_TOPICS  ## token coverage across all topics
# token_stats['n_per_doc'] = token_stats.id_size / token_stats.id_nunique ## token frequency / doc
# token_stats['pct_tokens'] = token_stats.id_size / token_stats.id_size.sum() ## % of all token occurrences
# token_stats['rank'] = token_stats.id_size.rank(ascending=False)

# what do we have now



######################## DTM
##
## Now lets take the data and put it back into a DTM 
## Keep only the top 500 tokens
##
## we will use the long dataformat, not the summary frame above

# the tokens in our summary frame

# TOP_N = 1000
# top_toks = (token_stats
#             .sort_values("id_size", ascending=False)
#             .iloc[:TOP_N, :][['token', "id_size", "pct_tokens"]])
# top_toks

# how many tokens and % of all words total
# PCT_TOTAL = top_toks.pct_tokens.sum()

# print(f"The top {TOP_N} tokens account for {PCT_TOTAL} all tokens logged.  \nIf 10 docs have 10 words, this is freq(token)/10*10")

# back to a dtm

# limit to the top N from above
# TOP_N_TOKENS = list(top_toks.token.values)

# # keep just the data that we need, in steps
# tl2 = tl.loc[tl.token.isin(TOP_N_TOKENS), :]
# tl3 = tl2.loc[:, ["pos", "token", "id"]]

# # the construction
# dtm = tl3.pivot_table(columns="token", 
#                       values="pos",  # just to summarise
#                       index=tl2.id,  # the document id from the original dataset
#                       aggfunc=np.count_nonzero)  # how many times did it appear per doc id

# what do we have



######################################  Quick question for you
## what are your observations about above?
##
##

## replace missing values

############ NOTE:
## 
## Truthfully, this a sparse represenation and one where we are adding space just til have a full matrix.  
## But lets just start here, because this format allows us to apply ML tasks!

#################################### Lets predict the category!
##
## we now have a dataset that can be used to fit a ML model.  
## the quality of the models and how we think about ML tasks is all about the data
## let's start with this framing for intuition
##
##

# imports - violating my rule of the thumb

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import metrics

# we need to get the label per doc
# we were given a long dataset, so lets not assume
# just groupby and get the data

# assumption that id and topic always exist, so just need to isolate the values

# labels = (tl
#           .loc[:, ["id", "topic"]]
#           .drop_duplicates()
#           .topic
#           .to_list())

# lets build the datasets for the model
# X = dtm.copy()
# y = labels

# confirm we have the same thing

# split the data

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=820)

# fit the model

# tree = DecisionTreeClassifier(max_depth=5, min_samples_split=30, min_samples_leaf=15)
# tree.fit(X_train, y_train)

# fit metrics on test

# preds = tree.predict(X_test)
# ctable = metrics.classification_report(y_test, preds)
# print(ctable)

# confusion matrix from skplot
# can see where the model isn't sure

# skplot.metrics.plot_confusion_matrix(y_test, preds, 
#                                      figsize=(7,4), 
#                                      x_tick_rotation=90 )

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

## review the slides at the end of this module
## predict spam
## objetive =  based on accuracy
## only input is text, but you can derive features
## limited time, but how do you maximize your time (and the model?)
## HINTS:
##        start small, simple models
##        iterate and see how you do against the leaderboard
##        code above helps you with the core mechanics
