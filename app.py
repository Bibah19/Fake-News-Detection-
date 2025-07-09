import streamlit as st
import joblib

# Load the vectorizer and model
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Streamlit UI
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is fake or real.")

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip():  # If there is input text
        transform_input = vectorizer.transform([news_input])  # Transform the input
        prediction = model.predict(transform_input)  # Predict using the model

        if prediction[0] == 1:  # Real news prediction
            st.success("The News is Real!")
        else:  # Fake news prediction
            st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to analyze.")