# imports 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# create some fake data
a = np.random.randint(0, 11, size=(25, 2))
adf = pd.DataFrame(a, columns=["x", "y"])


# lets summarize
summary = adf.x.value_counts()
summary.sort_values(ascending=False).plot(kind="bar")
plt.show()


# a plot 
adf.head(3)



# another plot
sns.scatterplot(data=adf, x="x", y="y")
plt.show()


# what just happened
plt.clf()



# how to handle it
sns.scatterplot(data=adf, x="x", y="y")
plt.show()
plt.clf()

summary.sort_values(ascending=False).plot(kind="bar")
plt.show()

# get some help

help(np.random.randint)
