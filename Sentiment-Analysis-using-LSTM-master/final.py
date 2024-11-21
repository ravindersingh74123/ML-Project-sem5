
import pickle as pk
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = pk.load(open('model-final.pkl', 'rb'))
tokenizer = pk.load(open('scaler-final.pkl', 'rb'))

# Streamlit UI for sentiment analysis
st.title("Sentiment Analysis Using LSTM")
st.write("This application predicts the sentiment of a given text input as **Positive** or **Negative**.")

# User input
user_input = st.text_area("Enter a sentence to analyze its sentiment:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        # Preprocess and predict
        twt = tokenizer.texts_to_sequences([user_input])
        twt = pad_sequences(twt, maxlen=30, dtype='int32', value=0)  # Adjust maxlen based on your model's input shape
        sentiment = model.predict(twt, batch_size=1, verbose=0)[0]

        # Display sentiment
        if np.argmax(sentiment) == 0:
            st.error("Sentiment: Negative")
        elif np.argmax(sentiment) == 1:
            st.success("Sentiment: Positive")
    else:
        st.warning("Please enter a sentence to analyze.")
