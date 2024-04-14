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

for word, score in top_words_with_scores:
    #print(f"Word: {word}, TF-IDF Score: {score}")
    print(word)


# anchor_words = [
#    ['sound', 'battery', 'charge', 'power', 'tv', 'light', 'mouse', 'keyboard', 'camera', 'case', 'lens', 'cable', 'speakers', 'quality', 'usb'], # Product Features and Quality

#    ['star', 'five', 'four', 'three', 'two', 'one'], # Quantitative Assessments
   
#    ['price', 'money', 'purchase', 'budget', 'pay', 'cheap', 'expensive', 'value', 'cost'], # Price and Value
   
#    ['good', 'great', 'nice','fantastic', 'better', 'love', 'poor', 'disappoint', 'excellent', 'bad', 'problem', 'issue', 'break', 'worst'], # Customer Opinions and Sentiment
   
#    ['work', 'buy', 'use', 'get', 'well', 'need', 'easy', 'expect', 'recommend', 'perfect', 'return', 'try', 'fit'] # Usage and Experience
# ]

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

topic_model = ct.Corex(n_hidden=10, max_iter = 12000, seed=2)
topic_model.fit(doc_word, words=words, anchors=anchor_words, anchor_strength = 15)






# Assuming 'topic_model' is your trained CorEx model
topics_with_scores = [topic_model.get_topics(topic=n, n_words=10) for n in range(topic_model.n_hidden)]
def filter_topic_words(topic_words_scores, threshold=0.1):
    """
    Filters words in a topic based on a mutual information score threshold.

    :param topic_words_scores: List of tuples containing (word, score, ...).
    :param threshold: Score threshold for filtering.
    :return: Filtered list of words.
    """
    # Adjust the unpacking to handle all returned values correctly
    return [word for word, score, *_ in topic_words_scores if score >= threshold]

# Extract topics and their scores
topics_with_scores = [topic_model.get_topics(topic=n, n_words=-1) for n in range(topic_model.n_hidden)]

# Apply filtering to each topic
threshold = 0.03  # Define your threshold
filtered_topics = [filter_topic_words(topic, threshold) for topic in topics_with_scores]

topic_names=['Product Performance and Reliability', 'Customer Satisfaction and Sentiment', 'Usability and Convenience', 'Price and Value for Money', 'Product Features and Specifications','Product Aesthetics and Design','Product Durability and Longevity', 'Customer Service and Warranty', 'Ease of Setup and Installation', 'Connectivity and Compatibility'   ]
# Print the filtered topics
for i, topic in enumerate(filtered_topics):
    print(f"Topic: {topic_names[i]}: {', '.join(topic)}")





topic_words_all = []
for n in range(len(anchor_words)):
    topic_words,_,_ = zip(*topic_model.get_topics(topic=n))
    topic_words_all.append(list(topic_words))
    print('{}: '.format(n) + '{}: '.format(topic_names[n]) + ', '.join(topic_words))
print(f'correlation: {topic_model.tc}')


# Predict anchor topics for new sentences
# Assuming 'new_sentences' is a list of new sentences to predict topics for

# new_sentences = ["These are fantastic speakers for the price.  I purchased the RTi4 instead of the A1's.  They are identical in all respects except some exterior detailing."]

reviews=[['Absolutely love the screen clarity and the battery life is beyond expectation.'],
 ['The camera quality could be better. Pictures come out blurry in low light.'],
 ['Excellent phone for the price. Good build quality and feels solid in hand.'],
 ["Battery life is disappointing. Doesn't last a full day with moderate use."],
 ['Impressed with the fast charging. Gets to 100% in just about an hour.'],
 ['The user interface is a bit complicated for beginners. Takes time to get used to.'],
 ['Great phone but it overheats quickly during games or heavy use.'],
 ['The fingerprint sensor is very responsive. Unlocks the phone instantly.'],
 ['Camera is fantastic! Takes great photos, even in challenging lighting conditions.'],
 ['Sound quality from the speakers is mediocre. Expected better at this price point.'],
 ['Love the design and the lightweight feel. Very comfortable to hold.'],
 ['Had issues with connectivity. Drops Wi-Fi and Bluetooth connections frequently.'],
 ['Comes with too much bloatware. Spent hours removing unwanted apps.'],
 ['Battery life is a game changer. Easily lasts two days with normal use.'],
 ['The screen is prone to scratches. Recommend getting a screen protector immediately.'],
 ['Very satisfied with the purchase. A great balance of features and price.'],
 ['Disappointed with customer service. Had a hard time getting my issue resolved.'],
 ['The phone is water-resistant. Survived a drop in the pool without any issues.'],
 ['Storage capacity is excellent. Plenty of room for apps, photos, and videos.'],
 ['The phone is too slippery without a case. Dropped it a couple of times already.'],
 ['Face recognition works well, even in dimly lit environments.'],
 ['Not happy with the color accuracy of the screen. Looks a bit off.'],
 ['Phone charges slowly despite the fast charger. Not sure what the issue is.'],
 ['Excellent for gaming. Smooth performance and no lag even on high settings.'],
 ['The phone is quite bulky. Not easy to use with one hand.'],
 ['Software updates are frequent and improve the phone\'s performance significantly.'],
 ['The included headphones are of low quality. Better to buy your own.'],
 ['Happy with the dual SIM feature. Very convenient for travel.'],
 ['Camera zoom capabilities are impressive. Great for capturing distant subjects.'],
 ['The glossy finish attracts fingerprints and smudges. Needs constant cleaning.'],
 ['Signal reception is strong. Haven\'t experienced any call drops.'],
 ['Wish it had a headphone jack. Using the adapter is a hassle.'],
 ['The screen\'s brightness is excellent. Visible even under direct sunlight.'],
 ['Setup was a breeze. Transferred all my data from the old phone without any issues.'],
 ['The voice assistant is very accurate and useful for hands-free operations.'],
 ['Not satisfied with the night mode camera. Photos still look grainy.'],
 ['The curved screen edges are more of a gimmick. Accidental touches are common.'],
 ['Happy with the phone\'s performance. Handles multitasking without any slowdowns.'],
 ['The phone gets regular security updates, which is reassuring.'],
 ['The absence of expandable storage is a drawback. Have to rely on cloud services.'],
 ['The vibration motor is weak. Often miss calls because I don\'t feel it.'],
 ['Sturdy build. The phone survived a few drops without any damage.'],
 ['The matte finish on the back looks and feels premium.'],
 ['The display is stunning. Watching movies and videos is a pleasure.'],
 ['The phone is quite expensive. Not sure it\'s worth the high price.'],
 ['Gesture navigation is intuitive and easy to use once you get the hang of it.'],
 ['Call quality is clear, with no background noise or interference.'],
 ['The phone supports wireless charging, which is super convenient.'],
 ['The AI camera features are hit or miss. Sometimes they enhance photos, sometimes not.'],
 ['Satisfied with the privacy features. Feels secure using this phone.'],
 ['The large screen is great for reading and browsing, but makes the phone bulky.'],
 ['Fast processor. Apps launch quickly and run smoothly.'],
 ['The facial recognition feature struggles in low light conditions.'],
 ['Battery charging speed decreases significantly after a few months of use.'],
 ['The notification LED is missing. Hard to know if I have missed calls or messages.'],
 ['The haptic feedback feels very realistic. Adds a nice touch to interactions.'],
 ['The phone\'s performance is good, but it gets warm with prolonged use.'],
 ['Lack of a physical home button takes some getting used to.'],
 ['The always-on display is a useful feature. Easy to check time and notifications.'],
 ['Camera\'s portrait mode is amazing. Blurs the background beautifully.']]





def preprocess_with_filtered_words(document, filtered_words):
    """
    Preprocess a document to only include specific filtered words.

    :param document: The document to preprocess.
    :param filtered_words: The list of words to retain in the document.
    :return: The preprocessed document.
    """
    tokens = document.lower().split()
    filtered_tokens = [token for token in tokens if token in filtered_words]
    return ' '.join(filtered_tokens)
# Flatten the reviews list to work with them more easily
flattened_reviews = [review[0] for review in reviews]
all_filtered_words = set(word for topic in filtered_topics for word in topic)
# Assuming all_filtered_words is a set of all filtered words you want to retain
preprocessed_reviews = [preprocess_with_filtered_words(review, all_filtered_words) for review in flattened_reviews]
# Vectorize the preprocessed reviews
X_new = vectorizer.transform(preprocessed_reviews)
# Predict the topics
predicted_topics = topic_model.predict(X_new)
# Print the review text along with the names of the predicted topics
for i, topic_presence in enumerate(predicted_topics):
    # Extracting the topic names based on the presence (True) or absence (False)
    predicted_topic_names = [topic_names[j] for j, present in enumerate(topic_presence) if present]
    
    # Joining the topic names into a string to display
    predicted_topics_str = ', '.join(predicted_topic_names)
    
    # Printing the review and its associated topics
    if predicted_topics_str:  # Check if there's at least one topic predicted
        print(f"Review: {flattened_reviews[i]}\nPredicted Topics: {predicted_topics_str}\n")
    else:
        print(f"Review: {flattened_reviews[i]}\nPredicted Topics: None\n")



# for i in range(0,len(reviews)):
#     X_new = vectorizer.transform(reviews[i])
#     predicted_topics = topic_model.predict(X_new)
#     print(predicted_topics)





#Total correlation plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16);



# Calculating coherence score for each topic abd ploting the coherennce scores
documents = df['processed_reviewText'].tolist()
tokenized_documents = [doc.split() for doc in documents]
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
# Create a Gensim dictionary from the tokenized documents
dictionary = Dictionary(tokenized_documents)
# Create a Gensim corpus using the dictionary
corpus = [dictionary.doc2bow(text) for text in tokenized_documents]
topic_coherence_scores = []
for i, topic in enumerate(filtered_topics): #can also use topic_words_all in place of filtered_topics
    # Calculate coherence for this topic
    coherence_model_topic = CoherenceModel(topics=[topic], texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
    coherence_score_topic = coherence_model_topic.get_coherence()
    topic_coherence_scores.append(coherence_score_topic)
    print(f"Coherence Score for Topic {i+1} (C_V): {coherence_score_topic}")
# calculate the average coherence across all topics
average_coherence = sum(topic_coherence_scores) / len(topic_coherence_scores)
print(f"Average Coherence Score (C_V): {average_coherence}")

topic_labels = [f'Topic {i+1}' for i in range(len(filtered_topics))] #we can also use topic_names for labels
coherence_df = pd.DataFrame({'Coherence Score': topic_coherence_scores}, index=topic_labels)

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(coherence_df[['Coherence Score']], annot=True, cmap='Reds', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Coherence Score'})
plt.title('Coherence Scores of Topics')
plt.ylabel('Topics')
plt.show()
