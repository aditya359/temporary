# Import required libraries
import pandas as pd
import numpy as np
import nltk
# Download essential resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # Required for some lemmatization features
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset (replace with your file paths)
real_news = pd.read_csv('/True.csv')[['text']]
fake_news = pd.read_csv('/Fake.csv')[['text']]

# Add labels and combine datasets
real_news['label'] = 0  # 0 for real news
fake_news['label'] = 1  # 1 for fake news
df = pd.concat([real_news, fake_news]).sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return ' '.join(stemmed_words)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Split dataset
X = df['processed_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = clf.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Prediction function
def detect_fake_news(new_text):
    processed_text = preprocess_text(new_text)
    vectorized_text = tfidf.transform([processed_text])
    prediction = clf.predict(vectorized_text)
    return "Fake News" if prediction[0] == 1 else "Real News"

# Example usage
sample_news = "New study reveals chocolate helps lose weight instantly!"
print(f"\nPrediction: {detect_fake_news(sample_news)}")