# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px    # interactive plotting for notebook environments

from mlxtend.frequent_patterns import apriori, association_rules

# my BILLING project in Google Cloud - replace "questrom" with your project
PROJECT = "questrom"

# groceries = pd.read_csv("groceries.csv")

# big query
SQL = """
select * from `questrom.datasets.groceries`
"""

#groceries = pd.read_gbq(SQL, PROJECT)
groceries = pd.read_csv('/Users/yuxuanmei/Documents/GitHub/BA820_ym/lecture-notes/groceries.csv')

# groceries = pd.read_csv("datasets/groceries.csv")

type(groceries)
groceries.shape

groceries.sample(3)

# quick exercise
# how many unique transactions
# how many unique items
# are there any duplicates
groceries.sample(3)
groceries.tid.nunique()
groceries['tid'].nunique()

groceries.item.nunique()

dupes = groceries.duplicated()
dupes.sum()

groceries.head(3)


## lets build out the transactions datset
groceries['purchased'] = True
tx = groceries.pivot(index="tid", columns="item", values='purchased')
tx.shape

tx.info()
pd.options.display.max_rows=200
tx.info(verbose=True)
tx.isna().sum()

tx.fillna(False, inplace=True)

## lets confirm no missing values
tx.isna().sum().sum()

## quick peak into the data
tx.shape
tx.iloc[:5, :5]


# item frequency - items per transaction
item_freq = tx.sum(axis=1)
item_freq[:5]

item_freq.value_counts()
item_freq.value_counts().plot(kind="bar")
plt.show()

# quick exercise
# identify the top 25 items in the dataset
prod_freq = tx.sum(axis=0)
prod_freq[:5]
prod_freq.sort_values(ascending=False)[:25]
prod_freq.sort_values(ascending=False)[-10:]


# lets build some rules
itemsets = apriori(tx, min_support=.005, use_colnames=True)

# lets explore the itemsets
type(itemsets)
itemsets.head(3)

# lets apply the rules engine
rules = association_rules(itemsets, metric='confidence', min_threshold=.2)
type(rules)
rules.columns
rules.sample(1).T


# summarize
rules.describe()


# filter rules
water = rules.loc[rules.consequents=={'bottled water'}, :]
water.shape
water.head(10)
