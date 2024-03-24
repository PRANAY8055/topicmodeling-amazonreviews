import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Generate bigrams using Phrases model
    phrases = Phrases([tokens], min_count=1, threshold=1)
    bigram = Phraser(phrases)
    tokens = bigram[tokens]
    
    # Join tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

# Load data
df = pd.read_csv('./electronics_sample.csv')
df.drop('summary', axis=1, inplace=True)
df.dropna(subset=['reviewText'], inplace=True)
#df = df[:19700]

# Apply preprocessing to reviewText
df['processed_reviewText'] = df['reviewText'].apply(preprocess_text)
processed_df = df[['processed_reviewText']]

# Save the DataFrame to a new CSV file
processed_df.to_csv('processed_reviews.csv', index=False)
