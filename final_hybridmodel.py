from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

# Split the data with 2000 rows for testing and the rest for training
train_df, test_df = train_test_split(df, test_size=4000, random_state=42)

# Continue with the vectorization and TF-IDF calculation on the training data
def tokenization(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['processed_reviewText'])
    words = vectorizer.get_feature_names_out()
    mean_tfidf_scores = tfidf_matrix.mean(axis=0)
    sorted_indices = mean_tfidf_scores.argsort()[0, ::-1]
    sorted_indices = np.asarray(sorted_indices).flatten()
    top_words_indices = sorted_indices[:100]
    top_words = [words[idx] for idx in top_words_indices]
    top_tfidf_scores = [mean_tfidf_scores[0, idx] for idx in top_words_indices]
    top_words_with_scores = list(zip(top_words, top_tfidf_scores))
    return top_words, top_words_with_scores, vectorizer, tfidf_matrix

top_words, top_words_with_scores, vectorizer, tfidf_matrix = tokenization(train_df)

for word, score in top_words_with_scores:
    print(f"Word: {word}, TF-IDF Score: {score}")

# Optionally, you can save the processed training and testing datasets to CSV files
train_df.to_csv('./processed_electronics_reviews_train.csv', index=False)
test_df.to_csv('./processed_electronics_reviews_test.csv', index=False)




import scipy.sparse as sparse
from corextopic import corextopic as ct
# anchor_words = [
#     ['work', 'last', 'break', 'reliable', 'durable', 'faulty', 'malfunction', 'defect'],  # Product Performance and Reliability
#     ['good', 'great', 'excellent', 'bad', 'poor', 'disappoint', 'satisfy', 'happy', 'unhappy'],  # Customer Satisfaction and Sentiment
#     ['easy', 'use', 'convenient', 'simple', 'difficult'],  # Usability and Convenience
#     ['price', 'cheap', 'expensive', 'value', 'money', 'worth', 'affordable'],  # Price and Value for Money
#     ['quality', 'sound', 'battery', 'power', 'screen', 'resolution', 'storage', 'capacity'],  # Product Features and Specifications
#     ['look', 'design', 'style', 'appearance', 'aesthetic', 'build', 'material'],  # Product Aesthetics and Design
#     ['last', 'durable', 'longevity', 'wear', 'tear', 'break', 'maintenance'],  # Product Durability and Longevity
#     ['return', 'warranty', 'service', 'support', 'refund', 'exchange'],  # Customer Service and Warranty
#     ['setup', 'install', 'ready', 'configure', 'assembly', 'manual', 'guide'],  # Ease of Setup and Installation
#     ['connect', 'compatible', 'wireless', 'plug', 'interface', 'sync']  # Connectivity and Compatibility
# ]

anchor_words = [
    ['price', 'cost', 'money', 'buy', 'cheap'],
    ['sound', 'speakers', 'headphones', 'radio', 'audio'],
    ['camera', 'video', 'image', 'light', 'quality'],
    ['battery', 'charge', 'power', 'speed', 'warranty'],
    ['cord', 'usb', 'cable', 'port', 'hdmi', 'plug'],
    ['ship', 'weeks'],
    ['return', 'warranty', 'service', 'support', 'refund', 'exchange']
]



doc_word = sparse.csr_matrix(tfidf_matrix)
doc_word.shape # n_docs x m_words
words = list(np.asarray(vectorizer.get_feature_names_out()))

topic_model = ct.Corex(n_hidden=7, max_iter = 12000, seed=2)
topic_model.fit(doc_word, words=words, anchors=anchor_words, anchor_strength = 6)


topic_names=['price', 'audio', 'camera','battery','peripherals','shipping','customer service']
topic_words_all = []
for n in range(len(anchor_words)):
    topic_words,_,_ = zip(*topic_model.get_topics(topic=n))
    topic_words_all.append(list(topic_words))
    #printing top10 words from each topic
    print('{}: '.format(n) + '{}: '.format(topic_names[n]) + ', '.join(topic_words))
print(f'correlation: {topic_model.tc}')



#calculating coherence scores
# Assuming df['processed_reviewText'] contains your preprocessed documents
documents = train_df['processed_reviewText'].tolist()
tokenized_documents = [doc.split() for doc in documents]
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
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
# Optionally, if you want to calculate the average coherence across all topics
average_coherence = sum(topic_coherence_scores) / len(topic_coherence_scores)
print(f"Average Coherence Score (C_V): {average_coherence}")
# calculating u_mass coherence score
coherence_model_umass = CoherenceModel(topics=[topic], texts=tokenized_documents, dictionary=dictionary, coherence='u_mass')
total_coherence_umass = coherence_model_umass.get_coherence()
print(f"Total Coherence Score (U_Mass): {total_coherence_umass}")

import seaborn as sns
# Create a DataFrame for the heatmap
topic_labels = [f'Topic {i+1}' for i in range(len(topic_words_all))]
coherence_df = pd.DataFrame({'Coherence Score': topic_coherence_scores}, index=topic_labels)
# Plotting the heatmap
plt.figure(figsize=(4, 8))
sns.heatmap(coherence_df[['Coherence Score']], annot=True, cmap='Reds', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Coherence Score'})
plt.title('Coherence Scores of Topics')
plt.ylabel('Topics')
plt.show()



#printing first 10 reviews categorized by the model from test_data
# Assuming the 'vectorizer' and 'topic_model' are already fitted and 'topic_names' is defined
from collections import defaultdict
# Prepare the data
reviews = test_df  # Assuming test_df is defined and preprocessed correctly
flattened_reviews = [review for review in reviews['processed_reviewText']]
# Vectorize the reviews
X_new = vectorizer.transform(reviews['processed_reviewText'].apply(preprocess_text).tolist())
predicted_topics = topic_model.predict(X_new)
# Initialize a dictionary to store reviews by topic
topic_reviews = defaultdict(list)
# Store reviews in dictionary by topic
for i, topics in enumerate(predicted_topics):
    for topic_idx, present in enumerate(topics):
        if present:
            topic_reviews[topic_names[topic_idx]].append(flattened_reviews[i])
# Restrict to first 10 reviews per topic
for topic in topic_reviews:
    topic_reviews[topic] = topic_reviews[topic][:10]
# Print reviews for each topic
for topic in topic_names:
    print(f"Topic: {topic}")
    if topic_reviews[topic]:
        for review in topic_reviews[topic]:
            print(f"Review: {review}\n")
    else:
        print("No reviews for this topic.\n")



# #Normal prediction code
# for i, topic_presence in enumerate(predicted_topics):

#     predicted_topic_names = [topic_names[j] for j, present in enumerate(topic_presence) if present]
#     predicted_topics_str = ', '.join(predicted_topic_names)
    
#     if predicted_topics_str:
#         print(f"Review: {flattened_reviews[i]}\nPredicted Topics: {predicted_topics_str}\n")
#     else:
#         print(f"Review: {flattened_reviews[i]}\nPredicted Topics: None\n")
