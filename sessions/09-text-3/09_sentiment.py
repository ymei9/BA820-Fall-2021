##############################################################################
## Foundations in Text Analytics - Sentiment Analysis
##
## Learning goals:
##                 - reinforce text as a datasource
##                 - python packages for handling our corpus for these specific tasks
##                 - Sentiment analysis 
##                 - Sentiment analysis via ML
##                 - Build your own sentiment classifier!
##############################################################################

# installs
# ! pip install newspaper3k
# ! pip install wordcloud
# ! pip install emoji
# ! pip install nltk
# ! pip install scikit-plot
# ! pip install umap-learn
# ! pip install afinn
# ! pip install -U spacy
# ! pip install spacytextblob

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot

# some "fun" packages
from wordcloud import WordCloud
import emoji

import re

# text imports

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from afinn import Afinn

from newspaper import Article

##################################### let's get warmed up!
##
## we will be using the Big Query questrom.datasets.airline-intents
##
## select all the data from the table but only keep rows where the intent values are 
## atis_airfare, atis_ground_service, atis_airline, atis_abbreviation
##
## tokenize the data, and apply tfidf weighting
## use UMAP to reconstruct the dtm to 2 embeddings
## visualize the dataset via a scatterplot, and overlay the intent as a color on the plot
## Does UMAP help us sort the intents?





###################################### NLTK parsing
###################################### Quick highlight that there are pre-built tools!
## 
## we may not have to reinvent the wheel!
## NLTK has some built in tooling we can leverage!
## and trust me, other toolkits have their own approaches too!

# from nltk.tokenize import word_tokenize, RegexpTokenizer, WordPunctTokenizer, TweetTokenizer

# we may also need to download a tool to help with (sentence) parsing amongst other tasks

# nltk.download('punkt')

# corpus = ['I want my MTV! www.mtv.com', "Can't I have it all for $5.00 @customerservice #help"]

# want to zoom in on a tokenizer to help with twitter, and perhaps other social data
# social = TweetTokenizer()

# tokens_social = []
# for doc in corpus:
#   tokens_social.append(social.tokenize(doc))


# # what do we have
# tokens_social



##################################### Sentiment 1
##
##  We will start basic = word/dictionary-based approach
## 
## IDEA:  each word gets a score, sum up the score, thats it!
## it's intuitive, easy to explain, and customizable
## 
## Afinn - 2011
## https://github.com/fnielsen/afinn
##
## TLDR
## limited language support, but highlights an important concept
## build our dictionary, and score
## could even be emoticons!
## https://github.com/fnielsen/afinn/tree/master/afinn/data
##
##

# setup the afinn "model"

# let's just start with something basic

# let's try another

############### Question:  What do you notice?  What happened (outside of getting a score)?

## let's look at the data behind this
# URL = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt"
# ad = pd.read_csv(URL, sep='\t', header=None, names=['token', 'score'])
# ad.head(4)

# summary

# what are the values for score

## THOUGHT Exercise:  What stands out relative to the distribution and the summary stats?

# we can inspect easily to wrap our heads around the words

# another search

# let's go back to a statement, see the score, and break it down

# what is the list of floats being used for the score

# confirming that we can make ths more concrete, and that
# the other words are not considered important for sentiment
# in a lookup approach

##################################### YOUR TURN
##
##  there is a table on big query
##  datasets.bruins_twitter
##
##  get the records where the hour is 0,1,2,3
##  this is not a select *, you have to filter records
##  - TRICKY: apply afinn sentiment to each record
##  - ensure that the data sorted by status_id
##  - plot the sentiment score over the records (this is a timeseries - like view)
##  - calculate the average sentiment by hour
##
##



################## WHAT DO YOU NOTICE ABOVE
################## should these be slightly neutral or negative?



##################################### Quick departure
## word clouds
## you may have clients ask about this
## 
## let's break this down
##

# URL = "https://getthematic.com/insights/word-clouds-harm-insights/"
# article = Article(URL)
# article.download()
# article.parse()

# we will use the article text from here
# expects a string, more or less, not a list, per se
# wc = WordCloud(background_color="white")
# wordcloud = wc.generate(article.text)

# # Display the  plot:
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

#### run above a few times, look for
## 1- the color of word cloud
## 2- the placement relative of customer, visualization and capture

################################## THOUGHT EXPERIMENT/REVIEW OF ABOVE
##
## Ok, let's break this down
## we have seen distance measures + PCA/TSNE/UMPA
## in those situations, spatial placement matters
## general best practices - color should mean something too
## size in word clouds is fine
## Takeaway:  this is really, really "fun" way to explore data
##            but hardly one that should ever be considered anlaytics or findings (IN MY OPINION)
##
## WHY?   How does the shape of sunglasses help us understand the data? 
## https://www.littlemissdata.com/blog/wordclouds
##

##################################### Sentiment form TextBlob
## 
## NLP toolkit that goes beyond sentiment, so worth exploring
## we will see dive deeper into spacy on Wednesday, so this is a nice segue
## 
## The general framework: we operate on a document, not a corpus
## the document is parsed for a variety of NLP tasks
##
## POS, sentiment, noun-phrases, even spelling correction
## we are adding this as a spacy pipeline, which extends the concept or corpus level assessment
##

# setup -- isolate the flow

# import spacy
# from spacy import cli
# from spacytextblob.spacytextblob import SpacyTextBlob

# cli.download("en_core_web_sm")

# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe('spacytextblob')

# a simple corpus to show the document orientation

# we will talk about spacy later
# but this is iterating over the messages in the corpus effeciently
# spacy naturally wants to operate on a corpus

# lets look at the first


# we can get the sentiment polarity
# doc._.polarity

# the textblob toolkit also has a subjectivity estimator
# doc._.subjectivity

############################### REVIEW
# we can see above that we are still getting the objects, but we start to see some detail
#
# polarity = how positive/negative, which ranges from -1 to 1
# subjectivity = 0 -> 1, or fact -> opinion
#
# not shown above, but textblob's model also handles modifier words which we will see in a moment
#
# ALSO: the model?  Trained on reviews that were labeled (movie reviews), and also draws from other projects
#                   https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt
#                   https://github.com/sloria/TextBlob/blob/90cc87ab0f9e25f37379079840ec43aba59af440/textblob/en/sentiments.py
# Link below you can dive into the code for more view
#                   https://github.com/sloria/TextBlob/blob/eb08c120d364e908646731d60b4e4c6c1712ff63/textblob/_text.py

# lets go back to the results though, we can pull out the scores as needed
# sent = []
# for doc in corpus:
#   sent.append(TextBlob(doc).sentiment.polarity)

# sent

# NOTE:  that unlike afinn, these are not balanced, but perhaps because there is more than
# just a lookup

# lets try some other examples

###### you can see above that there are some smarts baked into this
##     textblob is able to look at the words and modify as needed, based on the intensity of the word (not shown)
#

# one other note
# Textblob ignores 1 letter words, just like sklearn does

# subjectivity?

##################################### Breakout
##
##  We will use the same bruins twitter dataset above
##  refer to above if you want to re-query the data
##  
## 
##  calculate the polarity (sentiment) and subjectivity for each tweet
##  create a scatterplot to evaluate the both metrics for the dataset
##
##   next plot the relationship between afinn score and textblob score
##









##################################### Other Approaches/ Discussion: Vader sentiment
## Very brief review
## There are other approaches that attempt the modified approach
## you should inspect them at a granular level to ensure that these work as you like
## 
## But for sake of completeness, lets see the output
##
## for a deeper review on compound scores
## https://stackoverflow.com/questions/40325980/how-is-the-vader-compound-polarity-score-calculated-in-python-nltk
##  -- sum of normalized scores, and is not directly related to the pos/negative/netural
##  https://github.com/cjhutto/vaderSentiment#about-the-scoring
##
## we get pos/neutral/negative distros 
## and a compound score which is what mostly used in practice
## > .05 = positive
## < -.05 = negative
## else neutral
## 
## like Textblob, its a model with modifiers and intensifiers
## but this model was trained on social media, so perhaps it may fit your data better
## AGAIN:  these tools help us get a report off the ground quickly, but we always should review
## 
###################### if you play around with above, and test it, it might just show that some of these rule
###################### approaches may not always be something we can use in production

##################################### API approach
## APIs from cloud services on Google

##################################### Finally, a domain-specific ML approach
## 
## Sometimes we need to roll our own
## 
## What does this mean?
## 1. collect a dataset
## 2. annotate the data with our own business rules
##  --------> Label studio?
## 3. we can use some of the tools above 
## ---------> generate a score, define a threshold, give labels
## ---------> fit a model on labels
## ---------> review, iterate, review, iterate
##
## Why build our own
## 
## - out of the box generalize (thats a theme you have heard me say)
## - domain specific words (for example, dataset above shows sports terms that are positive but not captured
## - also, sarcasm is hard to detect even with modifier approaches
##
##

# there is an airlines tweets dataset on biq query
# bring in questrom.datasets.airlines-tweets
# just the tweet_id, airline_sentiment, airline, and text columns





# what do we have for a label distribution?

# lets assume our excellent back office staff has labeled these datasets properly
# huge assumption, right!
# we will parse the tweets, convert emojis, keep top 1000 vocab
# and then create our own ML-based sentiment classifier

# example of demojizing a text - just parses them out!

# txt = "great , but I have to go with #CarrieUnderwood 😍👌"
# emoji.demojize(txt)

# from nltk.tokenize import TweetTokenizer
# import emoji

# SW = stopwords.words('english')

# def tokenizer(text):
#   social = TweetTokenizer()
#   # replace emojis with string representations
#   text = emoji.demojize(text)
#   # if two emojis are stacked, add a whitespace in between
#   text = text.replace("::", ": :")
#   return social.tokenize(text)







############# notice above, what could we do better?

# regardless lets fit a decision tree classifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split

# tree = DecisionTreeClassifier(max_depth=5, min_samples_split=40, random_state=820)

# validation


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y)

# fit the tree

# apply the tree

# the report

############################ how might you improve this?
## think about the pipeline that we using to fit a model
## YOUR TURN:  Fit a different classifier to the dataset.