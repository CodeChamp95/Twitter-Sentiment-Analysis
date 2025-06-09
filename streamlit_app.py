import streamlit as st
import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Load tokenizer and model architecture
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("CodeChamp95/bert_twitter_sentiment_tokenizer")
    model = TFAutoModelForSequenceClassification.from_pretrained("CodeChamp95/bert_twitter_sentiment_model")
    # model.load_weights("bert_sentiment_model/tf_model.h5")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# model.load_weights("CodeChamp95/bert_twitter_sentiment_model/tf_model.h5")

# Label map (customize to match your model)
label_map = {
    0: "Negative",
    1: "Positive"
}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    pred_class = np.argmax(probs)
    confidence = round(probs[pred_class] * 100, 2)
    return label_map[pred_class], confidence

# UI styling
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")
# App layout
st.markdown("<h1>🐦 Twitter Sentiment Analyzer </h1>", unsafe_allow_html=True)

tweet = st.text_area("✍️ Enter a tweet to analyze:",height=200)

if st.button("🔍 Analyze Sentiment"):
    if tweet.strip():
        sentiment, confidence = predict_sentiment(tweet)
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence}%")
    else:
        st.warning("Please enter a tweet.")
