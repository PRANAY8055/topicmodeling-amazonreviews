from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

import pandas as pd
import numpy as np
import nltk
import re

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('./electronics_sample.csv')
df.drop('summary', axis=1, inplace=True)
df.dropna(subset=['reviewText'], inplace=True)
stop_words = set(stopwords.words('english'))

stop_words.update(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "star"])

def preprocess_text(text):
    text = text.lower()
    
    tokens = text.split()
    
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    
    tokens = [word for word in tokens if word not in stop_words]
    
    text = ' '.join(tokens)
    
    text = re.sub(r'[^a-z\s]', '', text)  
    
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
    
    return top_words, top_words_with_scores
    
top_words, top_words_with_scores = tokenization()

print("Top 10 words with their TF-IDF scores:")
for word, score in top_words_with_scores:
    print(f"Word: {word}, TF-IDF Score: {score}")
    
    
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = [top_words]

word2vec_model = Word2Vec(words, vector_size=100, window=5, min_count=1, workers=4)

word_vectors = [word2vec_model.wv[word] for word in top_words]

pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

kmeans = KMeans(n_clusters=5)
kmeans.fit(word_vectors_2d)
cluster_labels = kmeans.labels_

plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Word Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Initialize an empty dictionary to store word clusters
word_clusters = {}

# Iterate over each word and its corresponding cluster label
for word, cluster_label in zip(top_words, cluster_labels):
    # Check if the cluster label already exists in the dictionary
    if f"cluster-{cluster_label + 1}" not in word_clusters:
        # If the cluster label doesn't exist, create a new list for it
        word_clusters[f"cluster-{cluster_label + 1}"] = [word]
    else:
        # If the cluster label already exists, append the word to its list
        word_clusters[f"cluster-{cluster_label + 1}"].append(word)

# Print the word clusters
for cluster, words in word_clusters.items():
    print(f"{cluster}: {words}")

df_clusters = pd.DataFrame(word_clusters.items(), columns=['Cluster', 'Words'])

df_clusters.to_csv('./clusters.csv', index=False)