from flask import Flask, request, render_template
import os
import regex as re
import streamlit as st
from pickle import load
import nltk
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the model and vectorizer
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/svm_classifier_C-1_deg-1_gam-scale_ker-linear_42.sav")
model = load(open(model_path, "rb"))

vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/tfidf_vectorizer.sav")
vectorizer = load(open(vectorizer_path, "rb"))

class_dict = {
    "1": "POSITIVE",
    "0": "NEGATIVE"
}

# Ensure wordnet and stopwords are downloaded once
#download("wordnet")
#download("stopwords")

try:
    stop_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

#stop_words = stopwords.words("english")
#stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove any character that is not a letter (a-z) or white space ( )
    text = re.sub(r'[^a-z ]', " ", text)

    # Remove white spaces
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\^[a-zA-Z]\s+', " ", text)

    # Multiple white spaces into one
    text = re.sub(r'\s+', " ", text.lower())

    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    return text.split()

def lemmatize_text(words):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    return tokens

# Streamlit app layout
st.title("Spotify - Model Prediction")
st.write("Enter the review below:")

# User input
val1 = st.text_input("Review:", "")

if st.button("Predict"):
    if val1:
        val1 = preprocess_text(val1)
        val1 = lemmatize_text(val1)

        # Join tokens back into a single string for vectorizer
        tokens_list = [" ".join(val1)]

        # Use the pre-trained vectorizer to transform the input
        val1 = vectorizer.transform(tokens_list).toarray()

        # Ensure the correct shape for model input
        prediction = str(model.predict(val1)[0])
        pred_class = class_dict[prediction]

        # Display prediction
        st.success(f"Prediction: {pred_class}")
    else:
        st.error("Please enter a review to make a prediction.")

if __name__ == "__main__":
    # Use the port provided by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Set host to 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port, debug=True)