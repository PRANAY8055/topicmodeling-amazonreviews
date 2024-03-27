# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:09:42 2024

@author: Sreekar
"""

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import gensim
from gensim import matutils, models
import scipy.sparse
from gensim import corpora, models
from collections import defaultdict
import operator
import pprint

from sklearn.feature_extraction.text import CountVectorizer
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

csv_path = r'C:\Users\Sreekar\Downloads\electronics_sample.csv\electronics_sample.csv' 
reviews_df = pd.read_csv(csv_path)

def advanced_preprocess(text):
    # Check if the input is a string, return an empty string if not
    if not isinstance(text, str):
        return ''
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)
reviews_df['processed_reviews'] = reviews_df['reviewText'].apply(advanced_preprocess)    

tokenized_reviews = [doc.split() for doc in reviews_df['processed_reviews']]

# Create a Gensim Dictionary from the tokenized reviews
id2word = corpora.Dictionary(tokenized_reviews)

# Create a Gensim Corpus from the Dictionary
corpus = [id2word.doc2bow(text) for text in tokenized_reviews]

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:10]]


# Aggregate word frequencies
word_freq = defaultdict(int)
for document in corpus:
    for word_id, freq in document:
        word_freq[word_id] += freq

total_docs = len(corpus)

normalized_word_freq = {word_id: freq / total_docs for word_id, freq in word_freq.items()}

sorted_normalized_word_freq = sorted(normalized_word_freq.items(), key=operator.itemgetter(1), reverse=True)

# Print the top 50 normalized word frequencies
for word_id, normalized_freq in sorted_normalized_word_freq[:50]:
    print(f"{id2word[word_id]}: {normalized_freq}")
      
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=10, passes=10, random_state=42)    

pprint.pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]    

# Define a threshold for topic assignment (e.g., 0.1 for 10% relevance)
threshold = 0.1
reviews_df['topics'] = [
    [topic_num for topic_num, prop_topic in lda_model.get_document_topics(corp) if prop_topic >= threshold]
    for corp in corpus
]
print(reviews_df[['reviewText', 'topics']].head())
