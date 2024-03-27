from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct

import pandas as pd
import scipy.sparse as ss
import numpy as np
import nltk
import re

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('./electronics_sample.csv')
df.drop('summary', axis=1, inplace=True)
df.dropna(subset=['reviewText'], inplace=True)
stop_words = set(stopwords.words('english'))
stop_words.update(["i've", "i'am", "i'm"])

# This is the first stage of the flow
# Pre-processing the text
def preprocess_text(text):
    text = text.lower()
    
    tokens = text.split()
    
    tokens = [word for word in tokens if word not in stop_words]
    
    text = ' '.join(tokens)
    
    text = re.sub(r'[^a-z\s]', '', text) 
    
    tokens = text.split()
    
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    
    text = ' '.join(tokens)
    
    return text

df['processed_reviewText'] = df['reviewText'].apply(preprocess_text)

df.to_csv('./processed_electronics_reviews.csv', index=False)

# We applied TfidfVectorizer to transform the text into a meaningful numerical representation. 
# The word with the high frequency in the document is given a high score.
def tokenization():
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_reviewText'])

    words = vectorizer.get_feature_names_out()

    mean_tfidf_scores = tfidf_matrix.mean(axis=0)

    sorted_indices = mean_tfidf_scores.argsort()[0, ::-1]
    sorted_indices = np.asarray(sorted_indices).flatten()
    
    top_words_indices = sorted_indices[0:100]
    top_words = [words[idx] for idx in top_words_indices]

    top_tfidf_scores = [mean_tfidf_scores[0, idx] for idx in top_words_indices]

    top_words_with_scores = list(zip(top_words, top_tfidf_scores))
    
    return top_words, top_words_with_scores, vectorizer, tfidf_matrix
    
top_words, top_words_with_scores, vectorizer, tfidf_matrix  = tokenization()

# From the top high frequency words we got from the TFIDF vectorizer
# we would generate the topics with each topic contains some anchor words
anchor_words = [
    ['work', 'last', 'break', 'reliable', 'durable', 'faulty', 'malfunction', 'defect'],  # Product Performance and Reliability
    ['good', 'great', 'excellent', 'bad', 'poor', 'disappoint', 'satisfy', 'happy', 'unhappy'],  # Customer Satisfaction and Sentiment
    ['easy', 'use', 'convenient', 'simple', 'difficult'],  # Usability and Convenience
    ['price', 'cheap', 'expensive', 'value', 'money', 'worth', 'affordable'],  # Price and Value for Money
    ['quality', 'sound', 'battery', 'power', 'screen', 'resolution', 'storage', 'capacity'],  # Product Features and Specifications
    ['look', 'design', 'style', 'appearance', 'aesthetic', 'build', 'material'],  # Product Aesthetics and Design
    ['last', 'durable', 'longevity', 'wear', 'tear', 'break', 'maintenance'],  # Product Durability and Longevity
    ['return', 'warranty', 'service', 'support', 'refund', 'exchange'],  # Customer Service and Warranty
    ['setup', 'install', 'ready', 'configure', 'assembly', 'manual', 'guide'],  # Ease of Setup and Installation
    ['connect', 'compatible', 'wireless', 'plug', 'interface', 'sync']  # Connectivity and Compatibility
]

doc_word = ss.csr_matrix(tfidf_matrix)
doc_word.shape # n_docs x m_words
words = list(np.asarray(vectorizer.get_feature_names_out()))

# Initialize a Corex topic model with 10 topics, maximum 12000 iterations.
# Fit the Corex model to the TF-IDF document-word matrix using words as vocabulary and anchor words
topic_model = ct.Corex(n_hidden=10, max_iter = 12000, seed=2)
topic_model.fit(doc_word, words=words, anchors=anchor_words, anchor_strength = 15)

topics_with_scores = [topic_model.get_topics(topic=n, n_words=10) for n in range(topic_model.n_hidden)]

def filter_topic_words(topic_words_scores, threshold=0.1):
    return [word for word, score, *_ in topic_words_scores if score >= threshold]

topics_with_scores = [topic_model.get_topics(topic=n, n_words=-1) for n in range(topic_model.n_hidden)]

threshold = 0.03  # Define your threshold
filtered_topics = [filter_topic_words(topic, threshold) for topic in topics_with_scores]

topic_names=['Product Performance and Reliability', 'Customer Satisfaction and Sentiment', 'Usability and Convenience', 'Price and Value for Money', 'Product Features and Specifications','Product Aesthetics and Design','Product Durability and Longevity', 'Customer Service and Warranty', 'Ease of Setup and Installation', 'Connectivity and Compatibility'   ]
for i, topic in enumerate(filtered_topics):
    print(f"Topic: {topic_names[i]}: {', '.join(topic)}")

# Prediction of reviews into the topics using the corex model

reviews = pd.read_csv('./reviews_test.csv')['reviews'].tolist()

# Function to preprocess reviews with filtered words
def preprocess_with_filtered_words(document, filtered_words):
    tokens = document.lower().split()
    filtered_tokens = [token for token in tokens if token in filtered_words]
    return ' '.join(filtered_tokens)

flattened_reviews = [review[0] for review in reviews]
all_filtered_words = set(word for topic in filtered_topics for word in topic)

preprocessed_reviews = [preprocess_with_filtered_words(review, all_filtered_words) for review in flattened_reviews]

X_new = vectorizer.transform(preprocessed_reviews)
predicted_topics = topic_model.predict(X_new)

for i, topic_presence in enumerate(predicted_topics):

    predicted_topic_names = [topic_names[j] for j, present in enumerate(topic_presence) if present]
    predicted_topics_str = ', '.join(predicted_topic_names)
    
    if predicted_topics_str:
        print(f"Review: {flattened_reviews[i]}\nPredicted Topics: {predicted_topics_str}\n")
    else:
        print(f"Review: {flattened_reviews[i]}\nPredicted Topics: None\n")
