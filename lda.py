# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:09:42 2024

@author: Sreekar
"""

import numpy as np
import pandas as pd
import nltk
import re
import gensim
import operator
import pprint
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing the Data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

reviews_df = pd.read_csv(r'.\electronics_sample.csv' )
if 'summary' in reviews_df.columns:
    df.drop('summary', axis=1, inplace=True)
reviews_df.dropna(subset=['reviewText'], inplace=True)
stop_words.update(["i've", "i'am", "i'm"])

def advanced_preprocess(text):
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

reviews_df['processed_reviews'] = reviews_df['reviewText'].apply(advanced_preprocess)
print(reviews_df['processed_reviews'])

from gensim import matutils, models
import scipy.sparse

tokenized_reviews = [doc.split() for doc in reviews_df['processed_reviews']]

# Create a Gensim Dictionary from the tokenized reviews
id2word = corpora.Dictionary(tokenized_reviews)

# Create a Gensim Corpus from the Dictionary
corpus = [id2word.doc2bow(text) for text in tokenized_reviews]

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:10]]



word_freq = defaultdict(int)
for document in corpus:
    for word_id, freq in document:
        word_freq[word_id] += freq

total_docs = len(corpus)

normalized_word_freq = {word_id: freq / total_docs for word_id, freq in word_freq.items()}

# Sort normalized word frequencies
sorted_normalized_word_freq = sorted(normalized_word_freq.items(), key=operator.itemgetter(1), reverse=True)

for word_id, normalized_freq in sorted_normalized_word_freq[:50]:
    print(f"{id2word[word_id]}: {normalized_freq}")

# basic LDA Model
ldamodel = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=7, passes=10, random_state=42)

# Tuned LDA Model with alpha and eta
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=7, passes=10, alpha='auto', eta='auto', random_state=42)    

pprint.pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

reviews_df['topic'] = [max(lda_model.get_document_topics(corp), key=lambda x: x[1])[0] for corp in corpus]
print(reviews_df[['reviewText', 'topic']].head())

# Define a threshold for topic assignment 
threshold = 0.1
reviews_df['topics'] = [
    [topic_num for topic_num, prop_topic in lda_model.get_document_topics(corp) if prop_topic >= threshold]
    for corp in corpus
]
print(reviews_df[['reviewText', 'topics']].head())


# Calculating Coherence score for both the models
coherence_model_lda_cv = CoherenceModel(model=ldamodel, texts=tokenized_reviews, dictionary=id2word, coherence='c_v')
coherence_lda_cv = coherence_model_lda_cv.get_coherence()
print('Basic LDA Coherence Score (C_v): ', coherence_lda_cv)
coherence_model_lda_umass = CoherenceModel(model=ldamodel, corpus=corpus, dictionary=id2word, coherence="u_mass")
coherence_lda_umass = coherence_model_lda_umass.get_coherence()
print('Basic LDA Coherence Score (U_mass): ', coherence_lda_umass)
coherence_model_lda_cv = CoherenceModel(model=lda_model, texts=tokenized_reviews, dictionary=id2word, coherence='c_v')
coherence_lda_cv = coherence_model_lda_cv.get_coherence()
print('Tuned LDA Coherence Score (C_v): ', coherence_lda_cv)
coherence_model_lda_umass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence="u_mass")
coherence_lda_umass = coherence_model_lda_umass.get_coherence()
print('Tune LDA Coherence Score (U_mass): ', coherence_lda_umass)
