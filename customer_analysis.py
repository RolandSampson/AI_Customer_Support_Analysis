#-------------------------------------- Project: Customer Support Trend Analysis ---------------------------------------#
# Goal: Analyze raw text data to classify customer sentiment and identify key trends.
# This methodology is designed to provide clean, domain-grounded data for AI model training.
# Code style reflects structure, detailed comments, and use of pandas/NLP libraries.

# --- 1. Imports and Setup ---
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style (as seen in your submitted notebooks)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# --- 2. Data Loading and Initial Review ---

# Load the raw customer data (using tweets.csv as raw support transcripts for this demonstration)
# NOTE: Make sure the 'tweets.csv' file is in the same directory as this Python script
df = pd.read_csv('tweets.csv')

print("--- Initial Data Load ---")
print(f"Total Rows: {len(df)}")
print(df.head(2))

# --- 3. Data Cleaning and Preprocessing (Essential for NLP Accuracy) ---

# Initialize stop words list (to remove common, non-meaningful words)
stop_words = set(stopwords.words('english'))

# Function to perform all text cleaning steps (based on your Assignment 2 tasks)
def clean_text(text):
    # Remove RT (Retweet tag)
    text = re.sub(r'RT[\s]+', '', text)
    # Remove hyperlinks (URLs)
    text = re.sub(r'https?:\/\/\S+', '', text)
    # Remove Mentions (@) and the word after it
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#) but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove punctuation signs
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase the text to standardize words
    text = text.lower()
    
    # Remove Stopwords and apply Lemmatization (using TextBlob for efficiency)
    words = TextBlob(text).words
    cleaned_words = [str(word.lemma) for word in words if word.lower() not in stop_words]
    
    # Return the final cleaned string
    return " ".join(cleaned_words)

# Apply the full cleaning function to the 'Tweets' column
df['cleaned_text'] = df['Tweets'].apply(clean_text)

print("\n--- Cleaning Complete. Sample Cleaned Text ---")
print(df[['Tweets', 'cleaned_text']].head(2))


# --- 4. Sentiment Analysis (Core AI Training Component) ---

# Function to get polarity score (-1.0 to 1.0)
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Add Polarity column to the DataFrame
df['polarity'] = df['cleaned_text'].apply(get_polarity)

# Function to classify the text into Positive, Negative, or Neutral sentiment
def get_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Add the final Sentiment classification column
df['Sentiment'] = df['polarity'].apply(get_sentiment)

print("\n--- Sentiment Analysis Complete ---")
print(df[['Tweets', 'Sentiment']].head(5))

# --- 5. Visualization and Reporting (The Final Deliverable) ---

# Task: Plot the distribution of sentiment (shows overall emotional state of data)
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df, palette=['red', 'gray', 'green'])
plt.title('Distribution of Customer Sentiment')
plt.show()

# Task: Find the 10 most common words in the Negative Sentiment group (Key Pain Points)
negative_words = ' '.join(df[df['Sentiment'] == 'Negative']['cleaned_text']).split()
negative_freq = pd.Series(negative_words).value_counts().head(10)

# Visualize the 10 most common words in Negative Sentiment
plt.figure(figsize=(10, 6))
sns.barplot(x=negative_freq.values, y=negative_freq.index, palette='Reds_d')
plt.title('Top 10 Most Common Words in Negative Sentiment (Customer Pain Points)')
plt.xlabel('Word Count')
plt.ylabel('Words')
plt.show()

print("\n--- Project Analysis Complete ---")
