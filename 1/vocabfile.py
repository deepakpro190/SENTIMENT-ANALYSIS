import pickle
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
nltk.download('punkt_tab')

# Path to your preprocessed dataset
DATASET_PATH = 'imdb_processed.csv'  # Update with actual path

# Load the preprocessed dataset (assuming it's a CSV file with a column 'review' for text)
# Modify the column name accordingly if different
df = pd.read_csv(DATASET_PATH)

# Assuming the reviews are in the 'review' column
reviews = df['processed'].tolist()

# Tokenize and build vocabulary
def build_vocab_from_data(text_data, max_vocab_size=10000):
    # Tokenize each review
    all_words = []
    for review in text_data:
        all_words.extend(word_tokenize(review.lower()))  # Tokenize and make lowercase
    
    # Count frequency of each word
    counter = Counter(all_words)
    
    # Sort words by frequency and take top `max_vocab_size` words
    vocab = sorted(counter, key=counter.get, reverse=True)[:max_vocab_size]
    
    # Add <PAD> token to vocab
    vocab = ['<PAD>'] + vocab

    # Create word-to-index and index-to-word mappings
    word2int = {word: idx for idx, word in enumerate(vocab, 1)}
    int2word = {idx: word for word, idx in word2int.items()}
    
    return word2int, int2word

# Build the vocabulary from the dataset
word2int, int2word = build_vocab_from_data(reviews)

# Save vocabulary to a file (pickle format)
with open('vocab.pkl', 'wb') as f:
    pickle.dump((word2int, int2word), f)

print("Vocabulary has been built and saved to 'vocab.pkl'")
