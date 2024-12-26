import re
import nltk
import pandas as pd
import pickle
import spacy
from tqdm import tqdm
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Download necessary resources only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

# Load the dataset
dataset = pd.read_csv('customer_reviews.csv', encoding='ISO-8859-1')

# Define a function to label sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Apply the function to your dataset
dataset['Sentiment'] = dataset['Text'].apply(get_sentiment) 

# Dropping duplicates in the dataset
dataset = dataset.drop_duplicates(subset=['Id', 'ProductId', 'Summary', 'Text'], keep='first')


# Updated text cleaning function
def doTextCleaning(review):
    review = re.sub('<.*?>', '', review)
    review = re.sub(r'[^\w\s]', '', review)
    review = review.lower()
    tokens = word_tokenize(review)
    negation_words = {"not", "never", "no", "none", "nobody", "nothing", "neither", "nor"}
    processed_tokens = []
    skip_next = False

    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue
        word = tokens[i]
        if word in negation_words and i + 1 < len(tokens):
            next_word = tokens[i + 1]
            combined_word = f"{word}_{next_word}"
            processed_tokens.append(combined_word)
            skip_next = True  # Skip the next word as it has been combined
        elif word not in set(stopwords.words('english')) or word in negation_words:
            processed_tokens.append(word)

    # Lemmatization using spaCy
    doc = nlp(" ".join(processed_tokens))
    lemmatized_tokens = [token.lemma_ for token in doc if token.is_alpha]

    # Joining the cleaned tokens back into a string
    return " ".join(lemmatized_tokens)


# Preprocess the reviews with a progress bar
corpus = []
print("Cleaning text data...")
for review in tqdm(dataset['Text'], desc="Processing Reviews"):
    cleaned_review = doTextCleaning(review)
    corpus.append(cleaned_review)

# Additional features (e.g., length of text)
dataset['Text_Length'] = dataset['Text'].apply(lambda x: len(x.split()))

# Use TF-IDF for feature extraction with enhanced settings
print("\nVectorizing text data...")
vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=10000)
X = vectorizer.fit_transform(corpus).toarray()

# Adding additional features to the feature set
X = pd.DataFrame(X)
X['Text_Length'] = dataset['Text_Length']

# Assign labels (assuming sentiment is the target column)
y = dataset['Sentiment']

# Handle class imbalance using SMOTE
print("\nHandling class imbalance...")
X = pd.DataFrame(X)
X['Text_Length'] = dataset['Text_Length']

# Ensure all column names are strings
X.columns = X.columns.astype(str)

# Handle class imbalance using SMOTE
print("\nHandling class imbalance...")
smote = SMOTE(random_state=42)

X_resampled = []
y_resampled = []

# Create a progress bar
for i in tqdm(range(len(y.unique())), desc="Applying SMOTE"):
    X_res, y_res = smote.fit_resample(X, y)
    X_resampled.append(X_res)
    y_resampled.append(y_res)

# Combine resampled data
X = pd.concat(X_resampled, ignore_index=True)
y = pd.concat(y_resampled, ignore_index=True)

X=X.drop(columns='Text_Length')  

# Split dataset into training and testing sets
print("\nSplitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train a Multinomial Naive Bayes model
print("\nTraining the Multinomial Naive Bayes model...")
classifier = MultinomialNB(alpha=0.5)  # Adjusted alpha as part of hyperparameter tuning
classifier.fit(X_train, y_train)

# Save the trained model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Predict the test set results
print("\nMaking predictions on the test set...")
y_pred = classifier.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




