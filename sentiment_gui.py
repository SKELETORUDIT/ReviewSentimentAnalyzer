import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import pickle
from sentiment_analysis import doTextCleaning  # Import cleaning function
import re

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Function to remove HTML tags
def remove_html_tags(text):
    return re.sub('<.*?>', '', text)

# Load and preprocess dataset
reviews = pd.read_csv('customer_reviews.csv', encoding='ISO-8859-1')
reviews['Text'] = reviews['Text'].apply(remove_html_tags)  

# Function to update feedback
def update_feedback(index):
    review_text = reviews['Text'].iloc[index]
    processed_review = doTextCleaning(review_text)
    X_review = vectorizer.transform([processed_review]).toarray()
    predicted_sentiment = model.predict(X_review)[0]

    emoji_map = {'Positive': happy_emoji, 'Neutral': neutral_emoji, 'Negative': angry_emoji}
    emoji_image = emoji_map.get(predicted_sentiment, neutral_emoji)

    emoji_label.config(image=emoji_image)
    emoji_label.image = emoji_image
    review_label.config(text=review_text)

# Initialize the app
root = tk.Tk()
root.title("Sentiment Analysis Feedback")

# Load emojis
happy_emoji = ImageTk.PhotoImage(Image.open('happy_emoji.png').resize((100, 100)))
neutral_emoji = ImageTk.PhotoImage(Image.open('neutral_emoji.png').resize((100, 100)))
angry_emoji = ImageTk.PhotoImage(Image.open('angry_emoji.png').resize((100, 100)))

# Create widgets
emoji_label = tk.Label(root)
emoji_label.grid(row=0, column=0, padx=10, pady=10)

review_label = tk.Label(root, wraplength=300, justify="center")
review_label.grid(row=1, column=0, padx=10, pady=10)

current_index = [0]

def next_review():
    current_index[0] = (current_index[0] + 1) % len(reviews)
    update_feedback(current_index[0])

next_button = ttk.Button(root, text="Next Review", command=next_review)
next_button.grid(row=2, column=0, padx=10, pady=10)

# Initial display
update_feedback(current_index[0])

# Start GUI
root.mainloop()
