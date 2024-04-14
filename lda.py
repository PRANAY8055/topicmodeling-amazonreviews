
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

csv_path = r'electronics_sample.csv' 
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



from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assume 'documents', 'tokenized_documents', and 'dictionary' are already defined
# Assume 'lda_model' is your trained LDA model
# If not defined, set up as shown in previous explanations

# Extracting the top words for each topic
top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.append([word for word, _ in lda_model.show_topic(t, topn=10)])

# Calculate the coherence score for each topic using the C_V measure
topic_coherence_scores = []
for top_words in top_words_per_topic:
    coherence_model_topic = CoherenceModel(topics=[top_words], texts=tokenized_reviews, dictionary=id2word, coherence='c_v')
    coherence_score_topic = coherence_model_topic.get_coherence()
    topic_coherence_scores.append(coherence_score_topic)
    print(f"Coherence Score for Topic {i+1} (C_V): {coherence_score_topic}")
average_coherence = sum(topic_coherence_scores) / len(topic_coherence_scores)
print(f"Average Coherence Score (C_V): {average_coherence}")
# Create a DataFrame for the heatmap
topic_labels = [f'Topic {i+1}' for i in range(len(top_words_per_topic))]
coherence_df = pd.DataFrame({'Coherence Score': topic_coherence_scores}, index=topic_labels)

# Plotting the heatmap using a red color palette
plt.figure(figsize=(4, 8))
sns.heatmap(coherence_df[['Coherence Score']], annot=True, cmap='Reds', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Coherence Score'})
plt.title('Topic-wise Coherence Scores for LDA Model')
plt.ylabel('Topics')
plt.show()



