import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Load your HR data (replace with your own dataset)
# Here we create a simple example dataset
data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Feedback': [
        "The work environment is fantastic, I love the team.",
        "I feel very stressed out and unsupported by management.",
        "Work-life balance is great, but I need more challenges.",
        "There are too many micromanagement issues.",
        "I enjoy my job, but I wish there were more growth opportunities."
    ],
    'Performance': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']  # Example of labeled data
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Streamlit App Header
st.title('HR Data Analytics - Sentiment Analysis')

# Display the raw data
st.subheader('Employee Feedback Data')
st.write(df)

# NLP Sentiment Analysis
st.subheader('Feedback Sentiment Analysis')

# Function to analyze sentiment
def analyze_sentiment(feedback):
    score = sia.polarity_scores(feedback)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the sentiment analysis to each feedback
df['Sentiment'] = df['Feedback'].apply(analyze_sentiment)

# Display the feedback and sentiment
st.write("Feedback Sentiment Analysis:")
st.write(df[['Feedback', 'Sentiment']])

# Visualizing the Sentiment distribution
sentiment_counts = df['Sentiment'].value_counts()
fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title="Sentiment Distribution")
st.plotly_chart(fig)

# Example of Building a Classifier (Optional, can be used for more advanced analysis)
st.subheader('Train a Classifier for Performance Prediction')

# Vectorize the text feedback using TfidfVectorizer
X = df['Feedback']
y = df['Performance']

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Test the classifier
y_pred = classifier.predict(X_test)

# Display classification report and confusion matrix
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# Interactive feature for users to input their own feedback
st.subheader('Enter Your Feedback')
user_feedback = st.text_area("Feedback Text:")

if user_feedback:
    user_sentiment = analyze_sentiment(user_feedback)
    st.write(f"Sentiment of your feedback: {user_sentiment}")

    # Predict the performance label based on the trained classifier
    user_feedback_vectorized = vectorizer.transform([user_feedback])
    user_performance = classifier.predict(user_feedback_vectorized)
    st.write(f"Predicted Performance: {user_performance[0]}")
