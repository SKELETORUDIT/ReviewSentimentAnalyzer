import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Load the saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Load and preprocess the dataset
reviews = pd.read_csv('customer_reviews.csv', encoding='ISO-8859-1')

# Preprocess function from sentiment_analysis.py
def preprocess_review(text):
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize the text
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word, 'v') for word in text if word not in set(stopwords.words('english'))]
    return " ".join(text)

# Preprocess the first review and predict sentiment
first_review = reviews['Text'].iloc[0]
processed_review = preprocess_review(first_review)
X_review = vectorizer.transform([processed_review]).toarray()
predicted_sentiment = model.predict(X_review)[0]

# Load emojis
happy_emoji = mpimg.imread('happy_emoji.png')
neutral_emoji = mpimg.imread('neutral_emoji.png')
angry_emoji = mpimg.imread('angry_emoji.png')

# Map sentiment prediction to emoji
emoji_map = {1: happy_emoji, 0: neutral_emoji, -1: angry_emoji}
emoji = emoji_map.get(predicted_sentiment, neutral_emoji)

# Create a figure to display the emoji and the review text
fig, ax = plt.subplots(figsize=(6, 4))

# Hide the axes
ax.axis('off')

# Display the emoji
ax.imshow(emoji, extent=[0.2, 0.8, 0.4, 1])

# Display the review text below the emoji
ax.text(0.5, 0.2, first_review, fontsize=12, ha='center', va='center', wrap=True, 
        bbox=dict(facecolor='lightyellow', edgecolor='none'))

# Show the plot
plt.tight_layout()
plt.show()

