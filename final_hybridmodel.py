from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct

import pandas as pd
import scipy.sparse as ss
import numpy as np
import nltk
import re
import seaborn as sns

import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

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


anchor_words = [
    ['price', 'cost', 'money', 'buy', 'cheap'],
    ['sound', 'speakers', 'headphones', 'radio', 'audio'],
    ['camera', 'video', 'image', 'light', 'quality'],
    ['battery', 'charge', 'power', 'speed', 'warranty'],
    ['cord', 'usb', 'cable', 'port', 'hdmi', 'plug'],
    ['ship', 'weeks'],
    ['return', 'warranty', 'service', 'support', 'refund', 'exchange']
]


doc_word = ss.csr_matrix(tfidf_matrix)
doc_word.shape # n_docs x m_words
words = list(np.asarray(vectorizer.get_feature_names_out()))

# Initialize a Corex topic model with 10 topics, maximum 12000 iterations.
# Fit the Corex model to the TF-IDF document-word matrix using words as vocabulary and anchor words
topic_model = ct.Corex(n_hidden=7, max_iter = 12000, seed=2, n_jobs=1)
topic_model.fit(doc_word, words=words, anchors=anchor_words, anchor_strength = 6)

topics_with_scores = [topic_model.get_topics(topic=n, n_words=10) for n in range(topic_model.n_hidden)]

topic_words_all = []

for topic in topics_with_scores:
    words = [word for word, score,_ in topic]
    topic_words_all.append(words)


# # Prediction of reviews into the topics using the corex model
# reviews = pd.read_csv('./processed_electronics_reviews_test.csv')

# flattened_reviews = [review for review in reviews['processed_reviewText']]

# preprocessed_reviews = reviews['processed_reviewText'].apply(preprocess_text).tolist()

# X_new = vectorizer.transform(preprocessed_reviews)
# predicted_topics = topic_model.predict(X_new)

# for i, topic_presence in enumerate(predicted_topics):

#     predicted_topic_names = [topic_names[j] for j, present in enumerate(topic_presence) if present]
#     predicted_topics_str = ', '.join(predicted_topic_names)
    
#     if predicted_topics_str:
#         print(f"Review: {flattened_reviews[i]}\nPredicted Topics: {predicted_topics_str}\n")
#     else:
#         print(f"Review: {flattened_reviews[i]}\nPredicted Topics: None\n")      

# Calculating coherence score for each topic abd ploting the coherennce scores
documents = df['processed_reviewText'].tolist()
tokenized_documents = [doc.split() for doc in documents]

# Create a Gensim dictionary from the tokenized documents
dictionary = Dictionary(tokenized_documents)
# Create a Gensim corpus using the dictionary
corpus = [dictionary.doc2bow(text) for text in tokenized_documents]
topic_coherence_scores = []
for i, topic in enumerate(topic_words_all):
    # Calculate coherence for this topic
    coherence_model_topic = CoherenceModel(topics=[topic], texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
    coherence_score_topic = coherence_model_topic.get_coherence()
    topic_coherence_scores.append(coherence_score_topic)
    print(f"Coherence Score for Topic {i+1} (C_V): {coherence_score_topic}")
    
# calculate the average coherence across all topics
average_coherence = sum(topic_coherence_scores) / len(topic_coherence_scores)
print(f"Average Coherence Score (C_V): {average_coherence}")

# calculating u_mass coherence score
coherence_model_umass = CoherenceModel(topics=[topic], texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
total_coherence_umass = coherence_model_umass.get_coherence()
print(f"Total Coherence Score (U_Mass): {total_coherence_umass}")

topic_names=['price', 'audio', 'camera','battery','peripherals','shipping','customer service']
topic_labels = [f'Topic {i+1}' for i in range(len(topic_words_all))] #we can also use topic_names for labels
coherence_df = pd.DataFrame({'Coherence Score': topic_coherence_scores}, index=topic_labels)

# Plotting the heatmap
plt.figure(figsize=(3, 8))
sns.heatmap(coherence_df[['Coherence Score']], annot=True, cmap='Reds', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Coherence Score'})
plt.title('Coherence Scores of Topics')
plt.ylabel('Topics')
plt.show()
