import streamlit as st
from utils import predict_sentiment

st.title("IMDB Movie Review Sentiment Analysis")

st.write("Enter a Movie Review to Classify it as Positive or Negative")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    sentiment, score = predict_sentiment(review=user_input)

    st.write(f"Review: {user_input}")
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {score}")
else:
    st.write("Please Enter a Movie Review")


