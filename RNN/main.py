import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

## Mapping of word index to words
word_index=imdb.get_word_index()
reversed_word_index={value:key for key,value in word_index.items()}

#Load saved model
model=load_model('simple_rnn_imdb.h5')

## Helper Functions
# function for decoding review
def decoded_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3,'?')for i in encoded_review])

## function to preprocess user_input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


## Prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction= model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

## Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")



st.write("Enter a movie review to classify it as positive or negative")

## User Input
user_input=st.text_area("Movie Review")
if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)

    ## Make prediction
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    ## Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write("Please Enter a movie review.")

