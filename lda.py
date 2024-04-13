# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:09:42 2024

@author: Sreekar
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import gensim
from gensim import corpora
from collections import defaultdict
import operator
import pprint

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

reviews_df = pd.read_csv('./electronics_sample.csv')
reviews_df.drop('summary', axis=1, inplace=True)
reviews_df.dropna(subset=['reviewText'], inplace=True)

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

reviews_df['processed_reviews'] = reviews_df['reviewText'].apply(preprocess_text)

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