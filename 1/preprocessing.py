import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer and stopwords
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning functions
def rm_link(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def rm_punct2(text):
    return re.sub(r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)

def rm_html(text):
    return re.sub(r'<[^>]+>', '', text)

def space_bt_punct(text):
    pattern = r'([.,!?-])'
    s = re.sub(pattern, r' \1 ', text)
    s = re.sub(r'\s{2,}', ' ', s)
    return s

def rm_number(text):
    return re.sub(r'\d+', '', text)

def rm_whitespaces(text):
    return re.sub(r' +', ' ', text)

def rm_nonascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)

def rm_emoji(text):
    emojis = re.compile(
        '['
        u'\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F1E0-\U0001F1FF'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE
    )
    return emojis.sub(r'', text)

def spell_correction(text):
    return re.sub(r'(.)\1+', r'\1\1', text)

def clean_pipeline(text):
    no_link = rm_link(text)
    no_html = rm_html(no_link)
    space_punct = space_bt_punct(no_html)
    no_punct = rm_punct2(space_punct)
    no_number = rm_number(no_punct)
    no_whitespaces = rm_whitespaces(no_number)
    no_nonasci = rm_nonascii(no_whitespaces)
    no_emoji = rm_emoji(no_nonasci)
    spell_corrected = spell_correction(no_emoji)
    return spell_corrected

# Preprocessing pipeline
def tokenize(text):
    return word_tokenize(text)

def rm_stopwords(tokens):
    return [i for i in tokens if i not in stopwords]

def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_pipeline(text):
    tokens = tokenize(text)
    no_stopwords = rm_stopwords(tokens)
    lemmas = lemmatize(no_stopwords)
    return ' '.join(lemmas)

# Vocabulary building and sequence conversion
def build_vocab_from_data(data, max_vocab_size):
    """
    Builds a vocabulary dictionary from the provided data.
    Each word will be assigned an index based on its frequency.
    """
    vocab = {}
    for line in data:
        for word in line.split():
            vocab[word] = vocab.get(word, 0) + 1
    
    # Sort vocab by frequency and limit to max_vocab_size
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
    
    # Create a word2index dictionary (mapping word -> index)
    word2int = {word: idx + 1 for idx, (word, _) in enumerate(sorted_vocab)}  # Adding +1 to avoid 0 index being used
    return word2int

def text_to_sequence(text, vocab):
    """
    Converts a cleaned and preprocessed text to a sequence of integers using the provided vocabulary.
    """
    if not isinstance(vocab, dict):
        raise TypeError("vocab should be a dictionary.")
    return [vocab.get(word, 0) for word in text.split()]
