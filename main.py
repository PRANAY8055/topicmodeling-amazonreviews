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

def tokenization():
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_reviewText'])

    words = vectorizer.get_feature_names()

    mean_tfidf_scores = tfidf_matrix.mean(axis=0)

    sorted_indices = mean_tfidf_scores.argsort()[0, ::-1]
    sorted_indices = np.asarray(sorted_indices).flatten()
    
    top_words_indices = sorted_indices[0:100]
    top_words = [words[idx] for idx in top_words_indices]

    top_tfidf_scores = [mean_tfidf_scores[0, idx] for idx in top_words_indices]

    top_words_with_scores = list(zip(top_words, top_tfidf_scores))
    
    return top_words, top_words_with_scores, vectorizer, tfidf_matrix
    
top_words, top_words_with_scores, vectorizer, tfidf_matrix  = tokenization()

for word, score in top_words_with_scores:
    print(f"Word: {word}, TF-IDF Score: {score}")
    
anchor_words = [
   [ 'sound', 'time', 'battery', 'back', 'charge', 'power', 'tv', 'light', 'mouse', 'keyboard', 'camera', 'case' , 
    'lens', 'cable', 'speakers', 'headphones'], # Product description

    ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'number', 'eight', 'nine', 'ten', 'count', 'quantity'], # Quantity

    ['price','cash',  'money', 'purchase',  'budget',  'pay', 'card'], # Price

    [ 'good','nice', 'better', 'look', 'love', 'break', 'problem', 'issue', 'great', 
     'disappoint', 'worst', 'trouble', 'poor'], # Opinion
    
    ['work','buy','well','need','time','little','really','expect',
     'much','perfect','easy','recommend','quality'] # Behaviour of the product
]

doc_word = ss.csr_matrix(tfidf_matrix)
doc_word.shape # n_docs x m_words
words = list(np.asarray(vectorizer.get_feature_names()))

no_of_topics = len(anchor_words)
anchor_strength = 5
topic_model = ct.Corex(n_hidden=no_of_topics, seed=2)
topic_model.fit(doc_word, words=words, anchors=anchor_words, anchor_strength=anchor_strength)

for n in range(len(anchor_words)):
    topic_words,_,_ = zip(*topic_model.get_topics(topic=n))
    print('{}: '.format(n) + ', '.join(topic_words))


# Predict anchor topics for new sentences
# Assuming 'new_sentences' is a list of new sentences to predict topics for

new_sentences = ['This product is really waste of money', 'Works well, I bought more than five times']

X_new = vectorizer.transform(new_sentences)
predicted_topics = topic_model.predict(X_new)
print(predicted_topics)



