import sys
sys.path.append("1")

import streamlit as st
import torch
from model import load_model
from preprocessing import clean_pipeline, preprocess_pipeline, text_to_sequence
import nltk
import pickle
import os

# Download necessary NLTK resource for tokenization
nltk.download('punkt')

# Constants for file paths
MODEL_PATH = "1/sentiment_lstm.pt"  # Path to your .pt file
VOCAB_FILE = "1/vocab.pkl"  # Path to the saved vocab file
VOCAB_SIZE = 121301  # Estimated vocab size used during training
device = torch.device('cpu')  # Use CPU for model inference

# Load model
model = load_model(MODEL_PATH, vocab_size=VOCAB_SIZE, device=device)

# Load saved vocabulary from file
def load_vocab(vocab_file):
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}! Ensure the file is in the correct location.")
    with open(vocab_file, 'rb') as f:
        word2int, int2word = pickle.load(f)  # Load the tuple (word2int, int2word)
    return word2int, int2word

# Load vocabulary
word2int, int2word = load_vocab(VOCAB_FILE)

# Streamlit UI
st.title("Movie Review Sentiment Analysis")

# Input for review
st.header("Enter a Movie Review")
user_input = st.text_area("Type or paste your movie review below:")

if st.button("Analyze Sentiment"):
    if not user_input:
        st.error("Please enter a review to analyze sentiment.")
    else:
        try:
            # Preprocess input
            cleaned_text = clean_pipeline(user_input)
            preprocessed_text = preprocess_pipeline(cleaned_text)

            # Convert to sequence using word2int from the loaded vocab
            sequence = text_to_sequence(preprocessed_text, word2int)
            sequence_tensor = torch.tensor([sequence])

            # Predict sentiment
            with torch.no_grad():
                output = model(sequence_tensor)
                
                # Extract prediction value and determine sentiment
                prediction = output.item()  # Extract the scalar value from the tensor
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                st.success(f"Predicted Sentiment: {sentiment}")
        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {e}")
