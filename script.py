import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "deepakpro190/sentiment-analysis-roberta"

# Load model and tokenizer directly from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# Load RoBERTa model and tokenizer from "Roberta" directory
#MODEL_DIR = "Roberta"
#tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
#model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cpu")  # Running on CPU

# Move model to device
model.to(device)
model.eval()  # Set model to evaluation mode

# Streamlit UI
st.title("Movie Review Sentiment Analysis with RoBERTa")

# Input for review
st.header("Enter a Movie Review")
user_input = st.text_area("Type or paste your movie review below:")

if st.button("Analyze Sentiment"):
    if not user_input:
        st.error("Please enter a review to analyze sentiment.")
    else:
        try:
            # Tokenize input using RoBERTa tokenizer
            inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to correct device

            # Predict sentiment
            with torch.no_grad():
                output = model(**inputs)
                logits = output.logits
                prediction = torch.argmax(logits, dim=1).item()  # Get class index

            # Convert prediction to sentiment
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.success(f"Predicted Sentiment: {sentiment}")

        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {e}")
