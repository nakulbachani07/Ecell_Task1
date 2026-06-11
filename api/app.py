import sys
sys.path.append(".")

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from src.preprocess import clean_text,truncate_text

# creat FastAPI app
app = FastAPI(
    title="SEC Filing Risk Classification API",
    description="API for predicting financial risk level from 10-K filing text",
    version="1.0"
)

#load saved models
model = joblib.load("models/xgboost_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

#define input format
class PredictionRequest(BaseModel):
    text: str

#Define home route
@app.get("/")
def home():
    return {
        "message": "SEC Filing Classification API is running.",
        "endpoint": "/predict"
    }

#define prediction route
@app.post("/predict")
def predict(request: PredictionRequest):
    #input text
    input_text = request.text

    #clean text
    cleaned_text = clean_text(input_text)

    #truncate text
    cleaned_text = truncate_text(cleaned_text, max_chars=10000)

    #Convert text to TF-IDF features
    X_input = vectorizer.transform([cleaned_text])

    # Predict numeric class
    prediction = model.predict(X_input)[0]

    #Convert numeric prediction to label
    label = label_encoder.inverse_transform([prediction])[0]

    # Get prediction confidence
    probabilities = model.predict_proba(X_input)[0]
    confidence = float(np.max(probabilities))

     # Return result
    return {
        "label": label,
        "confidence": round(confidence, 4)
    }

