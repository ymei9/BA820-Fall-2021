# create an endpoint to create a sentiment score using Afinn


from fastapitableau import FastAPITableau
from typing import List
from afinn import Afinn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

app = FastAPITableau(
    title="Simple Sentiment example",
    description="A *very basic* sentiment scoring app.  Caveat emptor.",
    version="1.0.0",
)


# setup the english "model"
afinn = Afinn(language='en')

@app.post("/sentiment",
    summary="Calculate a simple sentiment score for a piece of text",
    response_description="A sentiment score for the text.")
def sentiment(text: List[str]) -> List[float]:
    scores = [afinn.score(t) for t in text]
    return scores

