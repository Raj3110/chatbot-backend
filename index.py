import identity
import identity.web
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify, url_for, redirect, session, render_template
from flask_cors import CORS, cross_origin
from flask_dance.contrib.google import make_google_blueprint, google
from flask_jwt_extended import create_access_token, jwt_required, create_refresh_token, JWTManager, get_jwt_identity
from flask_session import Session
from msal import ConfidentialClientApplication
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()
from pathlib import Path
import json
import urllib
import msal
import random
import string
import base64
from flask_apscheduler import APScheduler
from email.message import EmailMessage
from flask_mail import Mail, Message

# from flask_oauthlib.client import OAuth

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)
# app.config['CORS_HEADERS'] = 'Content-Type'

# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
app.secret_key = os.urandom(12)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)
jwt = JWTManager(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()



@app.route("/")
def index():
    print('running.....')


# Apurv code - chatbot
from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Load the models
tfidf = load('models/tfidf.pkl')
rf = load('models/rf.pkl')
label_encoder = load('models/label_encoder.pkl')

if tfidf is None or rf is None or label_encoder is None:
    print("Error: Failed to load one or more models. Please check the pickle files and versions.")

# Load the CSV data
df = pd.read_csv('models/Chat_bot_02_oct.csv')  # Replace with your actual CSV file path

# Download NLTK data (if necessary)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Apostrophe expansion dictionary
Apostrophes_expansion = {
    "'s": " is",
    "'re": " are",
    "'r": " are",
    "n't": " not",
    "'ve": " have",
    "'d": " would",
    "'ll": " will",
    "dept.": "department"
}

# Function to expand apostrophes
def remove_apostrophes(text):
    """Expand common apostrophes."""
    words = word_tokenize(text)
    processed_text = [Apostrophes_expansion[word] if word in Apostrophes_expansion else word for word in words]
    return " ".join(processed_text)

# Stopwords removal
stop_words = stopwords.words('english')

def sw_rem(text):
    """Remove stopwords from text."""
    return ' '.join(word for word in text.split() if word not in stop_words)

from textblob import TextBlob  # For spelling correction
# Spelling correction using TextBlob
def correct_spelling(text):
    """Correct spelling in the text using TextBlob."""
    corrected_text = str(TextBlob(text).correct())
    return corrected_text

# Stemming
stems = SnowballStemmer('english')

def stemming(text):
    """Apply stemming to words in text."""
    return ' '.join(stems.stem(word) for word in text.split())

import re
def pipeline(text):
    """Apply text preprocessing steps: spelling correction, lowercase, punctuation removal, apostrophe expansion, stopwords removal, and stemming."""
    text = remove_apostrophes(text)  # Expand apostrophes
    #text = correct_spelling(text)  # Correct spelling
    text = str(text).lower()  # Convert to lowercase
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation

    #text = sw_rem(text)  # Remove stopwords
    text = stemming(text)  # Apply stemming
    return text

# Apply preprocessing to the questions in the dataset
df['N_Question'] = df['Question'].apply(pipeline)

def vectorization(text):
    vector = tfidf.transform([text]).toarray()
    return list(vector[0])

# Add vector column to the dataset
df['vector'] = df['N_Question'].apply(vectorization)

def query(input_text):
    input_text = pipeline(input_text)
    input_vector = tfidf.transform([input_text]).toarray()

    # Predict the category for the input text
    input_category = list(label_encoder.inverse_transform([rf.predict(input_vector)[0]]))[0]

    # Filter dataset for matching category
    df_cat = df[df['Category'] == input_category]

    # Get the similarity of the input with the available questions
    all_vec = np.array(df_cat['vector'].to_list())
    text_similarity = cosine_similarity(X=all_vec, Y=input_vector)

    # Sort and retrieve the top most similar answer
    answers = sorted(list(enumerate(text_similarity)), reverse=True, key=lambda x: x[1])

    # Check for similarity threshold
    if answers[0][1] <= 0.5:
        return "I don't have that information. Please contact helpdesk@aichefmaster.com"

    # Output the best match and ask for feedback
    answer = df_cat.iloc[answers[0][0]]['Answer']

    # Return the best match
    return answer

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']

    greetings = ["Hi there!", "Hello!", "Hie!!", "Hey!!"]
    # Initialize session variables if not present
    if user_message.lower() in ["hi", "hello", "hie", "hey"]:
        return jsonify({
            'reply': f"{np.random.choice(greetings)} How can I assist you,?",
            'is_greeting': True
        })

    # Check for exit condition
    if user_message.lower() in ['bye', 'goodbye', 'exit', 'ok bye']:
        goodbyes = ["Goodbye!", "Take care!", "Have a great day ahead!"]
        return jsonify({
            'reply': np.random.choice(goodbyes),
            'is_goodbye': True
        })

    response = query(user_message)

    return jsonify({'reply': response})



if __name__ == '__main__':
    app.run(debug=True)  # , host='127.0.0.2')
