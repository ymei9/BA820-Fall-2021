##############################################################################
## Fundamentals for pratical Text Analytics - spacy for NLP tasks
##                                            NER, 
##                                            use models to extract objects in our text
##                                            why:  reporting/summarization
##                                                  use downstream (API calls, db lookups)
##
##
## Learning goals:
##                 - SPACY!
##                 - NER extraction
##                 - annotating our own data is a real thing
##
##
## Great resources
##                 - https://spacy.io/usage/spacy-101
##
##############################################################################

# installs
! pip install -U spacy 
! pip install -U textacy
! pip install newspaper3k
! pip install afinn

# imports
import spacy
from spacy import cli
from spacy import displacy
import pandas as pd


# upcoming!
# from textacy.extract.keyterms import textrank
# import gensim

import textacy

from newspaper import Article
import json

from afinn import Afinn



"""# Spacy"""

######################################### lets get started with spacy!
##
## "industrial strength" NLP tools
## increasingly the go to in this space, but like Textblob, 
## it focuses on document-oriented operations
## lots of configuration and power, but lets get the foundations set
##

"""![](https://d33wubrfki0l68.cloudfront.net/3ad0582d97663a1272ffc4ccf09f1c5b335b17e9/7f49c/pipeline-fde48da9b43661abcdf62ab70a546d71.svg)"""

# If VS Code crashes
# use
# (from command line python -m spacy download en_core_web_md)
# and ignore the cli commands below, proceed to spacy.load

# model = "en_core_web_md"
# cli.download(model)

# nlp = spacy.load('en_core_web_md')

# spacy operates via language models
# https://spacy.io/models
# we setup the model as nlp, the spacy convention

# the message we will parse into a document


# we are using all of the defaults, but this is a "heavy" parse with a ton of great data to explore
# https://spacy.io/usage/spacy-101#features

# what do we have?



# we can list it, but those are not ordinary pieces of text!

# tokens

# the tokens have properties/attributes
# https://spacy.io/usage/linguistic-features#tokenization

"""![](https://spacy.io/tokenization-9b27c0f6fe98dcb26239eba4d3ba1f3d.svg)"""

# link to pseudocode for how spacy tokenizes 
# https://spacy.io/usage/linguistic-features#how-tokenizer-works

# what do we have with a token?
# lots of nlp elements!

# attributes
# https://spacy.io/api/token

# go back to the doc

# extacting the tokenization in spacy

# we saw above, each token is an object we can inspect
# remove punct


# make lower case

# remove stopwords
# good examples on accessing/modifying stopwords:
# https://stackoverflow.com/questions/41170726/add-remove-custom-stop-words-with-spacy

# [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]

# lets just keep everything as is

# parsed = [(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.head) for token in doc]

# ok, there is a lot going on, but we can now see the idea of the language models in action
# we are tokenizing text
# the text itself has representations, like pos, lemma, etc.
# https://spacy.io/usage/linguistic-features#pos-tagging

# luckily, spacy has built in help!

# spacy, via it's models (learned from labeled data!) can even map the relationships


# for those in jupyter or colab
# displacy.render(doc, style="dep", jupyter=True)

# if in vs code - spacy has a built in server
# the viz is at http://localhost:5000 
# displacy.serve(doc, style="dep")

#######################################  YOUR TURN
# using the corpus below, parse the text and POS tag, and lemma

# corpus = ["Brock likes golf and hockey",
#           "I love teaching at Questrom",
#           "Python makes data analytics fun!"]











"""![](https://hashouttech.com/static/3c5973a520b86c3660a9771453df5794/2bef9/span-object.png)"""

# before we dive into NER, its worth noting that docs are a brilliant representation for text
# at the core, and to keep simple, think like slicing elements of a list







# these are just, well, spans of a document
# think slices

# many of the same elements apply

# why is this important?
# a token, or span of tokens, could have meaning
# Named Entity Recognition - NER



"""# Named Entity Recognition (NER)"""

#######################################  Named Entity Recognition
## 
## We have seen regex can be very powerful
## not only can we tokenize data, but we COULD use it to parse patterns
##
## HOWEVER:  the spacy parsing has already trained a GENERALIZED model for us
##           lets start there! But note, based on certain tasks, spacy is near/at SOTA
## 
## https://spacy.io/usage/linguistic-features#named-entities
## 
## Why does this matter?
## - we have large corpora and want to extract the entities being discussed
## - think legal documents -  which people/organizations are involved
## - news organizations tagging/categorizing articles to compare across all articles
## - content recommendations - other texts including this entity/entities
## - customer support - which products/services are our customers reference in service requests
## - medical - illnesses or diseases per medical intake forms
## - hiring/scanning: skill detection, experience detection
## - product names/ids





# 'basic' python still applies

# we can grab these out

# these have properties/attributes

# remember explain?

# there is a visualizer too
# of course, we can visualize this.  spacy is the bees knees

# displacy.render(doc, style="ent", jupyter=True)

# or in vs code -- localhost:5000
# displacy.serve(doc, style="ent")

# NER isn't perfect as we just saw
# another example

# corpus = ["Let's go to the store", "I enjoy programming in the language go"]

# goents = []
# for d in corpus:
#   doc = nlp(d)
#   for ent in doc.ents:
#     goents.append((ent.text, ent.label_)) 

# goents





#######################################  YOUR TURN
##
## parse the article at the URL below
## trick: consider this a document, not a corpus
## extract the entities like we have above into a list
## visualize, if you can

URL = "https://www.lyrics.com/lyric/180684/Billy+Joel/We+Didn%27t+Start+the+Fire"

# HINT:  we have seen how easy the newspaper3k package makes extracting data from the web!
#









"""## Custom NER Model"""

################################## Custom NER
##
## we will build train our own NER model
##
##

# first, download the repo
# we are git cloning it into our session

# ! git clone https://github.com/Btibert3/BA820-Fall-2021.git

# read in the pre-annotated dataset

# path = "/content/BA820-Fall-2021/apps/labelstudio/ner1.json"
# with open(path, "r") as f:
#   a = json.load(f)

"""> NOTE:  This was annotated using Label Studio!  The task was to identify product ids when supplied by customers in service requests.  This can be used to lookup information from a database for the customer service rep and expedite the process."""

# needs to be in the form of 
# [ (doc, {“entities”: [ (start, end, label) ]}) ]
# this assumes one label per entity, but will put any
##
# label studio annotated NER training file in the shape that spacy needs!

# TRAIN_DATA = []
# for entry in a:
#   doc = entry.get('data').get('text')
#   result = entry.get("annotations")[0].get('result')
#   if len(result) == 0:
#     continue
#   entities = []
#   for r in result:
#     ent = r.get('value')
#     entities.append([ent.get('start'), ent.get('end'), ent.get('labels')[0]])
#   TRAIN_DATA.append([doc, {'entities': entities}])

# NOTE: this script is commonly found as the reference build the spacy file format

# import srsly
# import typer
# import warnings
# from pathlib import Path

# import spacy
# from spacy.tokens import DocBin

# def convert(lang: str, TRAIN_DATA, output_path: Path):
#     nlp = spacy.blank(lang)
#     db = DocBin()
#     for text, annot in TRAIN_DATA:
#       doc = nlp.make_doc(text)
#       ents = []
#       for start, end, label in annot["entities"]:
#         # print(start, end, label)
#         span = doc.char_span(start, end, label=label)
#         if span is None:
#             msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
#             #warnings.warn(msg)
#         else:
#             ents.append(span)
#       doc.ents = ents
#       db.add(doc)
#     db.to_disk(output_path)

# # this wrties the file to our session standard working directory
# convert("en", TRAIN_DATA, "train.spacy")

# spacy requires/STRONGLY prefers (3+) that we use the explicit config system
# admittedly this adds a huge layer of complexity to the abstraction framework
# I suspect that this will become both easier and more transparent shortly

# the cli way to setup the model config
# !python -m spacy init config --lang en --pipeline ner config.cfg --force

# copy the train.spacy as dev.spacy  -- not great, 
# but I didn't create enough labels for a dev set

# ! cp train.spacy dev.spacy

# tell spacy to train the model on our datasets and export the model

# ! python -m spacy train config.cfg                     \
#                         --output ./                    \
#                         --paths.train ./train.spacy    \
#                         --paths.dev ./dev.spacy        \
#                         --training.max_epochs 5        \
#                         --training.eval_frequency 10

# load in the custom model 
# remember, we only taught this model one thing 
# an NER model specific for one entity, a product id 

# custom_nlp = spacy.load("model-best")

# lets see this relative to our original NLP model

# msg = "Brock teaches at Boston University"
# nlp(msg).ents

# and now with the custom model
# custom_nlp(msg).ents

# this makes sense
# now, lets send a domain-specific message
# try 123423-23, ABQ12234
# play around with different reps, can see the product

# msg2 = "Is the product (1232432) available for overnight shipping?"
# [(ent.label_, ent.text) for ent in custom_nlp(msg2).ents]

# try 123423-23, ABQ12234
# play around with different reps, can see the product matters 
# with (), and doesnt like - in the product id

# but what about when we pass above into the PRETRAINED spacy model?
# [(ent.label_, ent.text) for ent in nlp(msg2).ents]

"""> If you attempt different values for the product above, you will see the generic spacy NER model flags the entity, but it can vary.  

__So what can we do?__

We _could_ add the new NER model into the pipeline.  WOrth noting, though, is that the order of the pipeline matters.  Naturally you would want to put this last, BUT, we saw that products are being flagged by the generic model.  If that is first in the pipeline, it will override our more accurately trained, Product Entitty.  This has all sorts of issues, but worth noting.

> This is a common problem when trying to think about updating models.  Models can forget.  One approach, and the one that the folks at explosion suggest is to train a net-new model, but one that blends a silver-standard dataset with the gold-standard dataset for the new entity.

- Create a training set from the existing model by passing in text and using the NER annotations as truth.  These are inferences, so considered "silver".
- Append to the dataset above the training data for the new entity
- train a completely new model, and use that.


> In a way, my endless rant on domain-specific applies here.



"""

###################################### YOUR TURN
##
## Using the dataset on Big Query `questrom.datasets.uber`
## extract the entities using spacy's learned NER model
## identify the top 5 locations (GPE)
## 
##







