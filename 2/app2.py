import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load GloVe embeddings
embedding_index = {}
with open('glove.6B.200d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Define constants
MAX_NB_WORDS = 112237  # Vocabulary size
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200

# Initialize the tokenizer with GloVe's vocabulary
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>')

# Populate tokenizer's word index directly from GloVe
word_index = {word: i + 1 for i, word in enumerate(embedding_index.keys()) if i < MAX_NB_WORDS}
tokenizer.word_index = word_index

# Prepare the embedding matrix
embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_NB_WORDS:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Define the model architecture
model = Sequential([
    Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Load the model weights
try:
    model.load_weights('weight.hdf5')
except Exception as e:
    print(f"Error loading weights: {e}")

# Define the prediction function
def predict_sentiment(text):
    if not text or text.isspace():
        return "Invalid input. Please provide a valid sentence."

    # Tokenize input
    sequences = tokenizer.texts_to_sequences([text])
    
    # Check for empty or invalid sequences
    if not sequences or not sequences[0]:
        return "Unable to process the input text. Please use different words or a valid sentence."
    
    # Filter out None values from the sequence
    sequences[0] = [token for token in sequences[0] if token is not None]
    
    # Check again after filtering
    if not sequences[0]:
        return "Input text could not be processed due to unsupported words."

    print(f"Generated Sequence: {sequences[0]}")  # Debugging

    # Pad sequences to the defined max length
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Predict sentiment
    prediction = model.predict(data)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    
    return f"{sentiment} (Confidence: {confidence:.2f})"


# Streamlit app layout
st.title('Sentiment Analysis App')
st.write('Enter a sentence to analyze its sentiment.')

user_input = st.text_area('Input Text')

if st.button('Analyze'):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')
    else:
        st.write('Please enter some text to analyze.')
