# simple run through of a streamlit app

import pandas as pd
import streamlit as st
import joblib 
import numpy as np

# read in the models
pca = joblib.load("pca.joblib")
cv = joblib.load("cv.joblib")
tree = joblib.load("tree.joblib")

st.title("Streamlit Basic Example - SMS Data Challenge")

st.markdown("---")

# Declare a form and call methods directly on the returned object
form = st.form(key='sms_form', clear_on_submit=True)
msg = form.text_input(label='Enter the example SMS message')
submit = form.form_submit_button(label='Submit')

st.markdown("---")

st.markdown("""

A spam example:

> WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
"""
)


st.markdown("---")

if submit:
    # parse the input and get a prediction label
    dtm = cv.transform([msg]).toarray()  
    pcs = pca.transform(dtm)
    pred = tree.predict(pcs) 

    # for the prediction, get the image
    label = str(pred[0])

    st.text(f"Prediction: {label}")

    # image urls
    urls = {'spam': 'https://th-thumbnailer.cdn-si-edu.com/sihbBqxfKtgbOQ8pmFNJbQxFBFA=/fit-in/1600x0/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer/a3/a5/a3a5e93c-0fd2-4ee7-b2ec-04616b1727d1/kq4q5h7f-1498751693.jpg',
            'ham': 'https://t3.ftcdn.net/jpg/02/91/82/80/360_F_291828082_OJrqZHCJ91e3jgp3hKQLMXcfRCuaAQNP.jpg'
            }

    # render the prediction as an image
    FALLBACK_URL = "https://cdn.shopify.com/s/files/1/1061/1924/products/2_large.png?v=1571606116"
    st.image(urls.get(label, "FALLBACK_URL"))