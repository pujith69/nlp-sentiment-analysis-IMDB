from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# load model and tfidf
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text

def basic_preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def advanced_preprocess(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# fastapi app
app = FastAPI()

# request body structure
class ReviewRequest(BaseModel):
    review: str

@app.get('/')
def home():
    return {'message': 'Sentiment Analysis API is running'}

@app.post('/predict')
def predict(request: ReviewRequest):
    # run through pipeline
    cleaned = clean_text(request.review)
    preprocessed = basic_preprocess(cleaned)
    lemmatized = advanced_preprocess(preprocessed)
    vector = tfidf.transform([lemmatized])
    
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    
    return {
        'review': request.review,
        'sentiment': prediction,
        'confidence': round(float(probability.max()), 2),
        'neg_probability': round(float(probability[0]), 2),
        'pos_probability': round(float(probability[1]), 2)
    }