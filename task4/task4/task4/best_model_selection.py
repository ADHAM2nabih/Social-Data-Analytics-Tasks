import os
import io
import re
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

print("Starting Comprehensive Model Comparison (Evaluation & Error Analysis)...\n")

SCRIPT_DIR = Path(__file__).resolve().parent


def find_tasks_root(start_dir: Path) -> Path:
    for candidate in [start_dir] + list(start_dir.parents):
        if (candidate / "task3").is_dir() and (candidate / "task4").is_dir():
            return candidate
    return start_dir


TASKS_ROOT = find_tasks_root(SCRIPT_DIR)
DATA_PATH = TASKS_ROOT / "task3" / "final_data" / "labels" / "labeled_dataset.csv"
MODELS_DIR = SCRIPT_DIR / "models"
DOCS_DIR = SCRIPT_DIR / "docs"

# Store results
leaderboard = []

def add_result(category, model, scheme, feature, accuracy, precision, recall, f1, roc_auc):
    leaderboard.append({
        "Category": category,
        "Model": model,
        "Scheme": scheme,
        "Features": feature,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

# Load the ground truth data from Task 3
data_path = DATA_PATH
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"[!] Critical Error: Cannot find labeled data at {data_path}")
    exit(1)

# Ensure no NaNs in text and label, standardize labels
for col in ['v1_basic_text', 'v2_no_stop_text', 'v3_stem_text']:
    if col in df.columns:
        df[col] = df[col].fillna("")

df['final_label'] = df['final_label'].astype(str).str.strip().str.capitalize()

# Extract Targets
y_raw = df['final_label'].values
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Save Label Encoder for API Context
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
print(f"[INFO] LabelEncoder saved to {MODELS_DIR / 'label_encoder.pkl'}")

# Define Data Schemes dynamically based on our available dataset text outputs
schemes = {
    "v1_basic": df["v1_basic_text"].values,
    "v2_no_stop": df["v2_no_stop_text"].values,
    "v3_stem": df["v3_stem_text"].values
}

# Machine Learning Candidates with Hyperparameter Grids for Optimization
ml_models = {
    "Multinomial Naive Bayes": (MultinomialNB(), {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}),
    "Gaussian Naive Bayes (PCA)": (GaussianNB(), {'var_smoothing': [1e-9, 1e-8, 1e-7]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]})
}

best_model_obj = None
best_vectorizer_obj = None
best_pca_obj = None
best_acc = 0.0

print("1. Evaluating and OPTIMIZING 18 Combinations (3 Models x 3 Schemes x 2 Text Representations)...")

vectorizations = {
    "Bag-Of-Words": CountVectorizer(max_features=2500),
    "TF-IDF": TfidfVectorizer(max_features=2500)
}

# Core iteration for evaluating all schemas just like the SAD Project reference
for scheme_name, texts in schemes.items():
    print(f"\n   -> Evaluating Scheme: {scheme_name}")
    
    for vec_name, vectorizer in vectorizations.items():
        print(f"      -> Representation: {vec_name}")
        X_vec = vectorizer.fit_transform(texts).toarray()
        
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
        
        # Binarize labels for multi-class ROC-AUC
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        
        for model_name, (base_model, param_grid) in ml_models.items():
            try:
                grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

                if "PCA" in model_name:
                    pca = PCA(n_components=min(15, X_train.shape[1]), random_state=42)
                    X_train_pca = pca.fit_transform(X_train)
                    X_test_pca = pca.transform(X_test)
                    
                    grid_search.fit(X_train_pca, y_train)
                    best_model = grid_search.best_estimator_
                    preds = best_model.predict(X_test_pca)
                    if hasattr(best_model, "predict_proba"):
                        proba = best_model.predict_proba(X_test_pca)
                    else:
                        proba = None
                else:
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    preds = best_model.predict(X_test)
                    if hasattr(best_model, "predict_proba"):
                        proba = best_model.predict_proba(X_test)
                    else:
                        proba = None

                # Generate Required Task 4 Metrics
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                f1_val = f1_score(y_test, preds, average='weighted', zero_division=0)
                
                # ROC-AUC requires Probabilities
                roc_val = np.nan
                if proba is not None:
                    try:
                        roc_val = roc_auc_score(y_test_bin, proba, multi_class='ovr', average='weighted')
                    except ValueError:
                        pass
                
                if acc > best_acc:
                    best_acc = acc
                    best_model_obj = best_model
                    best_vectorizer_obj = vectorizer
                    if "PCA" in model_name:
                        best_pca_obj = pca
                    else:
                        best_pca_obj = None

                add_result("Machine Learning", f"{model_name} (Optimized)", scheme_name, vec_name, acc, prec, rec, f1_val, roc_val)
            except Exception as e:
                print(f"         [!] Error running {model_name} on {scheme_name} / {vec_name}: {e}")

# ==========================================
# 3. LEADERBOARD & SELECTION
# ==========================================
print("\n" + "="*80)
print("🏆 OVERALL LEADERBOARD 🏆")
print("="*80)

results_df = pd.DataFrame(leaderboard).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')
print(results_df.to_string())

# Save leaderboard for frontend
DOCS_DIR.mkdir(exist_ok=True)
results_df.to_csv(DOCS_DIR / "leaderboard.csv", index=False)

print("\n" + "="*80)
best_model_row = results_df.iloc[0]
print("🎉 WINNING CONFIGURATION (Deployed Object) 🎉")
print(f"Category: {best_model_row['Category']}")
print(f"Model:    {best_model_row['Model']}")
print(f"Scheme:   {best_model_row['Scheme']}")
print(f"Features: {best_model_row['Features']}")
print(f"Score:    {best_model_row['Accuracy']*100:.2f}% Test Accuracy")
print("="*80)

# Export winning stack exactly like expected in api.py
joblib.dump(best_model_obj, MODELS_DIR / "best_classifier_model.pkl")
joblib.dump(best_vectorizer_obj, MODELS_DIR / "vectorizer.pkl")
if best_pca_obj is not None:
    joblib.dump(best_pca_obj, MODELS_DIR / "pca_transformer.pkl")
else:
    # Wipe old pca if current winner doesn't use it
    pca_path = MODELS_DIR / "pca_transformer.pkl"
    if pca_path.exists():
        pca_path.unlink()

print("\n[INFO] Winning model objects and transformers exported to models/ directory successfully!")
print("[NEXT] Run 'python error_analysis.py' next to generate error matrix and CSVs.")
