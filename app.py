import nltk
nltk.download('vader_lexicon')  # NLTK data download at the start
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

# Initialize the SentimentIntensityAnalyzer with error handling
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

# Load HR data with more samples for better model training
data = {
    'EmployeeID': list(range(1, 11)),
    'Feedback': [
        "The work environment is fantastic, I love the team.",
        "I feel very stressed out and unsupported by management.",
        "Work-life balance is great, but I need more challenges.",
        "There are too many micromanagement issues.",
        "I enjoy my job, but I wish there were more growth opportunities.",
        "Excellent benefits and supportive colleagues.",
        "Constant pressure without proper resources.",
        "Good learning opportunities but poor work-life balance.",
        "Management needs to improve communication.",
        "Great company culture and flexible hours."
    ],
    'Performance': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive',
                    'Positive', 'Negative', 'Negative', 'Negative', 'Positive']
}

df = pd.DataFrame(data)

# Streamlit App Interface
st.title('HR Data Analytics - Sentiment Analysis')

# Display raw data with expander
with st.expander("View Raw Data"):
    st.write(df)

# Sentiment Analysis Section
st.subheader('Feedback Sentiment Analysis')

def analyze_sentiment(feedback):
    score = sia.polarity_scores(feedback)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    return 'Neutral'

df['Sentiment'] = df['Feedback'].apply(analyze_sentiment)

# Display sentiment results
col1, col2 = st.columns(2)
with col1:
    st.write("Feedback & Sentiment:", df[['Feedback', 'Sentiment']])
with col2:
    sentiment_counts = df['Sentiment'].value_counts()
    fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, 
                 title="Sentiment Distribution")
    st.plotly_chart(fig)

# Model Training Section
st.subheader('Performance Prediction Model')

# Enhanced data splitting with stratification
X = df['Feedback']
y = df['Performance']
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

# Handle small dataset splitting
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, 
        test_size=0.2, 
        random_state=42
    )

# Model training with error handling
if X_train.shape[0] > 0:
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Model evaluation
    if X_test.shape[0] > 0:
        y_pred = classifier.predict(X_test)
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
else:
    st.warning("Insufficient training data for model development")

# User Input Section
st.subheader('Feedback Analyzer')
user_feedback = st.text_area("Enter employee feedback:")

if user_feedback:
    # Sentiment analysis
    sentiment = analyze_sentiment(user_feedback)
    st.metric(label="Predicted Sentiment", value=sentiment)
    
    # Performance prediction
    if 'classifier' in locals():
        user_vec = vectorizer.transform([user_feedback])
        prediction = classifier.predict(user_vec)[0]
        st.metric(label="Predicted Performance", value=prediction)
    else:
        st.warning("Model not available for performance prediction")
