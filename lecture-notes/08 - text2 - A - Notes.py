# LEARNING GOALS
#
#                 - tokenization deeper dive
#                 - reinforce text prep and tokenization options
#                 - Cluster documents setup

# installs
# ! pip install newspaper3k
# ! pip install spacy
# ! pip install nltk
# ! pip install -U scikit-learn
# ! pip install scikit-plot
# ! pip install umap-learn
# ! pip install tokenwiser

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot

import re

# new imports
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk
from tokenwiser.textprep import HyphenTextPrep

from newspaper import Article

############################################ Get some data from the using an awesome package Newspaper3k!
## https://newspaper.readthedocs.io/en/latest/

# Boston based chatbot company, now called Mainstay
URL = "https://voicebot.ai/2021/02/16/conversational-ai-startup-admithub-raises-14m-for-higher-ed-chatbots/"

# # setup the article
article = Article(URL)

# # get the page
article.download()

# # parse it -- extracts all sorts of info
article.parse()

# what do we have -- b/c its for news sites, attempts to parse things like dates
article.publish_date

# the text -- what we are really after
article.text

# tokenize
cv = CountVectorizer()
atext = article.text

# sklearn expects iterables, like lists
atokens = cv.fit_transform([atext])

# how many tokens --- note the new syntax of get feature names out
len(cv.vocabulary_)
atokens.shape

# THOUGHT EXERCISE:
# we have a doc-term  matrix with one doc and the terms in the columns
# its effectively a 1d array
# hypothetically, if we had a reference database with the same term representation (vocabulary)
# what do you think we could do?

# IDEAS:  identify similar records (scipy cdist) or scikitlearn nearest neighbors



################################################## Lets summarize
##
## we can use sklearn to keep things in our typical ml format
## we can see that there is some pre-processing taking place
## lets dive into that a bit more, and then discuss a flow using nltk -> sklearn

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# - Notice the lower casing by default
# - we can pass our own regex/tokenizer if we wanted, and some people do this (build their own)
# - different ways to tokenize
# - there are stopwords, but we can pass anything
# - we can set the max number of tokens
# - we can one hot encode = instead of counts, it can be 0/1 for the word/token
# - we can create ngrams
# - we can even validate the vocabulary if we wanted
#
# This last point brings up the concept of unseen words
# Remember! sklearn fits the object, so any unseen words will not be parsed on new datsets with transform
#
# Summary: really powerful and adaptable, but means you plug in your own regex/tools

################### part 1: - lets start with ngrams
##
## instead of single tokens, we can try to capture context by windowing the tokens/phrases
## we can pass in a tuple of the ngrams, default is 1,1

# a new dataset
corpus = ["tokens, tokens everywhere"]

# we could only have bigrams
ngrams2 = CountVectorizer(ngram_range=(1,2))
ngrams2_tok = ngrams2.fit_transform(corpus)
ngrams2.vocabulary_

# the key point is that you can imagine it might be able to retain context
# if we combine tokens with other n-grams.  
#

###################################### Quick task
## 
## build off the chatbot article from above
## but instead of parsing the tokens (unigrams), include bigrams (2) and trigrams (3) 
## to the feature space
##
## how many features have we extracted from the article?
##

cv = CountVectorizer(ngram_range=(1,3))
atokens = cv.fit_transform([atext])
len(cv.vocabulary_)

###################################### Question
###### what does this say about our choice of tokenization
###### what tools might help with this "issue"?



###################################### Stopwords
## by default stop words are not removed
## there is a pre-built list of words, but let's ignore it
## nltk is a great toolkit, but for now
## lets just use the stopwords from that package

# if this is your first time, you may need to download the stopwords
# or on colab, for your session

nltk.download('stopwords')


## OF COURSE, you could always downlod your own.  not the format of below, we just pass in a list in the end

# lets get the stopwords
from nltk.corpus import stopwords
STOPWORDS = list(stopwords.words('english'))

# what do we have?
type(STOPWORDS)

# the first few
STOPWORDS[:5]

# note that everything is lower case!

# admittedly this is harder to find than it should be
# but the languages supported in NLTK

stopwords.fileids()

# now you can imagine that is pretty limiting above, I know
# the other approach is to use spacy
# https://spacy.io/usage/models
# we will dive into spacy later, but I think its important to keep building the intuition
# before going into model-driven work

# last, we can always add to the stoplist if we wanted to now that its a list abvoe

# lets keep the corpus small, so use the original article
# but remove stopwords

cv = CountVectorizer(stop_words=STOPWORDS)
atokens = cv.fit_transform([atext])
len(cv.vocabulary_) 

# 281 -> 237

# and of course, we can see the vocab
# cv.vocabulary_

###################################### Max tokens
## 
## this can be helpful if you want to restrict to the top N most frequent tokens
## this restricts your space at the start
## but the tradeoff is less common words, perhaps, could help with ML models
##     -- the tokens/phrases are specific to he known label, and while rare, often occur for the label

## we can use the article again, max with stopwords

# cv = CountVectorizer(max_features=20, stop_words=STOPWORDS)
# atokens = cv.fit_transform([atext])
# cv.vocabulary_

###################################### character tokens
## 
## if you wanted, you can parse characters
## a little out of scope, but highlighting the concept of tokenization can 
## take all sorts of forms!

x = ["Hello I can't"]
charvec = CountVectorizer(analyzer='char', ngram_range=(1,1))
char_tokens = charvec.fit_transform(x)
charvec.vocabulary_

charvec = CountVectorizer(analyzer='char', ngram_range=(2,7))
char_tokens = charvec.fit_transform(x)
charvec.vocabulary_

###################################### custom pattern
## 
## if you really wanted to (or needed to), you can roll your own
## tokenization
## This is a little forward looking  .....
## but highlights you all have the power to roll your own
##
## https://stackoverflow.com/questions/1576789/in-regex-what-does-w-mean
##

# alpha numeric plus a single quote/contraction
PATTERN = "[\w']+"

cv = CountVectorizer(token_pattern=PATTERN)
cv.fit(x)




###################################### Next Up:  Beyond simple counts with TFIDF
##
## instead of count vectors (which you can use, and should try, in your modeling!)
## we can try to de-prioritize common words 
## This surfaces words that may be less common, but nuanced and we want to prioritize those tokens
##

# the same data
corpus = ["Can't I have it all for $5.00 @customerservice #help", 
          'I want my MTV!']

# equivalent to CountVectorizer -> TfidfTransformer
# basically if you want tfidf, do this, it saves a step
# and you have the same options for parsing if you like

tfidf = TfidfVectorizer(token_pattern="[\w']+", ngram_range=(1,2))
tfidf.fit(corpus)

## just to call out, being able to specify the pattern can be 
## really powerful for specific tasks and business needs

# lets put this into a dataframe
idf = tfidf.transform(corpus)

idf = pd.DataFrame(idf.toarray(), columns=tfidf.get_feature_names_out())
idf


# we could even heatmap this to help understand the intuition here

plt.figure(figsize=(4,6))
sns.heatmap(idf.T, xticklabels=True, yticklabels=True, cmap='Reds')
plt.show()
