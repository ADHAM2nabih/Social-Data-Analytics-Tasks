from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
import os

warnings.filterwarnings('ignore')

# Download required NLTK data - ADDED punkt_tab which was missing
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) # Added explicit support for nltk >= 3.8+ tokenization changes
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass

app = Flask(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# -------------------------------------------------------------
# 1. Pipeline Cleaning
# -------------------------------------------------------------

def clean_text(text):
    if not isinstance(text, str):
        text = ""
        
    text = text.lower()
    # Simple regex cleanup
    import re
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    stopword_set = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopword_set]
    
    return " ".join(tokens)

# -------------------------------------------------------------
# 2. Initialization: Load Saved Models via Joblib
# -------------------------------------------------------------
print("[INFO] Loading saved models and transformers...")
try:
    classifier = joblib.load(MODELS_DIR / "best_classifier_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    
    pca = None
    pca_path = MODELS_DIR / "pca_transformer.pkl"
    if pca_path.exists():
        pca = joblib.load(pca_path)
        
    print("[INFO] All models loaded successfully!")
except Exception as e:
    print(f"[ERROR] Loading failed: {e}")
    print("[ERROR] Ensure you have run 'best_model_selection.py' first!")
    classifier, vectorizer, le, pca = None, None, None, None

# -------------------------------------------------------------
# 3. API Endpoints
# -------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None or vectorizer is None:
        return jsonify({"error": "Models are not loaded. Ensure pickle files exist."}), 500
        
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 1. Clean Text
    cleaned = clean_text(text)
    
    # 2. Vectorize Text
    vec_final = vectorizer.transform([cleaned]).toarray()
    
    # 3. PCA Transform (if applicable)
    if pca is not None:
        vec_final = pca.transform(vec_final)
    
    # 4. Predict
    pred_idx = classifier.predict(vec_final)[0]
    if hasattr(classifier, "predict_proba"):
        probas = classifier.predict_proba(vec_final)[0]
        confidence = float(np.max(probas))
    else:
        confidence = 1.0 # fallback for models lacking probas
    
    pred_class = le.inverse_transform([pred_idx])[0]
    
    return jsonify({
        "sentiment": str(pred_class),
        "confidence": round(confidence, 2),
        "cleaned_text_used": cleaned
    })

if __name__ == '__main__':
    # Run API on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
