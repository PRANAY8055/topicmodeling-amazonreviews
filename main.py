import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import re

nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv('./electronics_sample.csv')
df.drop('summary', axis=1, inplace=True)
df.dropna(subset=['reviewText'], inplace=True)

def preprocess_text(text):
    
    text = text.lower()
    
    text = re.sub(r'[^a-z\s]', '', text)  
    
    tokens = text.split()
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    text = ' '.join(tokens)
    return text

# Apply preprocessing to reviewText and summary
df['processed_reviewText'] = df['reviewText'].apply(preprocess_text)

# Save the DataFrame to an Excel file
df.to_csv('./processed_electronics_reviews.csv', index=False)
