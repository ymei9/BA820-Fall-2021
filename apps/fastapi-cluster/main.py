"""This is our app code that runs our API via FastAPI"""

from typing import Optional

from numpy.core.fromnumeric import shape
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from scipy.spatial.distance import cdist


# load the pickle file for our cluster definitions
with open("centers.pkl", "rb") as f:
    centers = pickle.load(f)

# create a dictionary to map the the clusters to a persona
persona = {0: 'High spenders, infrequent purchases',
           1: 'Brand loyalists',
           2: 'Low spenders, infrequence purchases',
           3: 'Weekend shoppers',
           4: 'Bargain Hungters'}

APP_DESC = """
##  Goals
- POC to highlight how a relatively flexible pipeline to train and serve ML models 
- Flag a model that can associate a customer with two attributes to a marketing segment
### Use cases
 - A new customer hits the `CRM`, and the `CRM` will post this data to an internal service (API) for customer segment definition
 - The segment can trigger automated marketing campaigns.
 - Also, note, that this documentation supports _markdown_.  How __awesome__ is that!
"""

# define our app
app = FastAPI(title="Example ML App to assign customers to a marketing persona/segment", 
              description=APP_DESC, 
              version="0.1")

# the base of the API, simple json return
@app.get("/")
def read_root():
    return {"Hello": "World"}

# define the structure of our dataset that we expect
# this helps us validate the data!
class Customer(BaseModel):
    orders: float
    spend: float
    cust_id: Optional[str] = None
    class Config:
        schema_extra = {
        "example": {
            'orders': 1.456,
            'spend': 5.67}
        }

@app.post("/persona/", tags=['marketing'])
async def get_persona(user: Customer):
    # This will use the customers data, pick the closest cluster center, and return
    # the persona segment.
    # Use case:  New data in a CRM (e.g. Salesforce), and this service tags the customer with a marketing segment
    
    # get the data from the POST to the api
    x, y = user.orders, user.spend
    cust = np.array([[x, y]]) 

    # use the data to find the closest cluster center
    reco_distance = cdist(cust, centers, metric='euclidean')
    segment = np.argmin(reco_distance)
    p = persona.get(segment) 

    # create the response - the dictionary will be served as json
    resp = {'persona': p, 'cust_id': user.cust_id}
    return resp









