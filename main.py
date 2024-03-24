
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv('./electronics_sample.csv')

# Handle missing values
df['reviewText'].fillna('missing', inplace=True)
df['summary'].fillna('missing', inplace=True)

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    # Tokenization
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    text = ' '.join(tokens)
    return text

# Apply preprocessing to reviewText and summary
df['processed_reviewText'] = df['reviewText'].apply(preprocess_text)
df['processed_summary'] = df['summary'].apply(preprocess_text)

# Show the processed data
print(df.head())
