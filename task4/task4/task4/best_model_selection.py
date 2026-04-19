import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.decomposition import PCA
import warnings
import json

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

# Reuse Task 3 lexical baselines (SentiWordNet + Bing Liu with negation)
LEXICAL_OK = False
try:
    TASK3_DIR = TASKS_ROOT / "task3"
    sys.path.insert(0, str(TASK3_DIR))
    from sentiment.lexical_models import (
        load_wordlist,
        sentiwordnet_style_predict,
        bing_liu_predict_with_negation,
    )

    POS_WORDS_PATH = TASK3_DIR / "words" / "positive-words.txt"
    NEG_WORDS_PATH = TASK3_DIR / "words" / "negative-words.txt"
    LEXICAL_OK = POS_WORDS_PATH.exists() and NEG_WORDS_PATH.exists()
except Exception:
    LEXICAL_OK = False

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
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            'n_estimators': [200, 400],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
        },
    ),
}

best_model_obj = None
best_vectorizer_obj = None
best_pca_obj = None
best_acc = 0.0
best_meta = {}

total_combos = len(ml_models) * len(schemes) * 2
print(f"1. Evaluating and OPTIMIZING {total_combos} Combinations ({len(ml_models)} Models x 3 Schemes x 2 Text Representations)...")

vectorizations = {
    "Bag-Of-Words": CountVectorizer(max_features=2500),
    "TF-IDF": TfidfVectorizer(max_features=2500),
}

# Core iteration for evaluating all schemas just like the SAD Project reference
for scheme_name, texts in schemes.items():
    print(f"\n   -> Evaluating Scheme: {scheme_name}")
    
    for vec_name, vectorizer in vectorizations.items():
        print(f"      -> Representation: {vec_name}")
        vec = clone(vectorizer)
        X_vec = vec.fit_transform(texts).toarray()
        
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
        
        # Binarize labels for multi-class ROC-AUC
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        
        for model_name, (base_model, param_grid) in ml_models.items():
            try:
                grid_search = GridSearchCV(clone(base_model), param_grid, cv=3, scoring='accuracy', n_jobs=-1)

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
                    best_vectorizer_obj = vec
                    if "PCA" in model_name:
                        best_pca_obj = pca
                    else:
                        best_pca_obj = None
                    best_meta = {
                        "scheme": scheme_name,
                        "features": vec_name,
                        "model_key": model_name,
                        "uses_pca": bool("PCA" in model_name),
                        "test_size": 0.2,
                        "random_state": 42,
                    }

                add_result("Machine Learning", f"{model_name} (Optimized)", scheme_name, vec_name, acc, prec, rec, f1_val, roc_val)
            except Exception as e:
                print(f"         [!] Error running {model_name} on {scheme_name} / {vec_name}: {e}")

# ------------------------------------------
# Lexical models (3 variants × 2 lexicons = 6)
# ------------------------------------------
if LEXICAL_OK:
    print("\n2. Evaluating Lexical Models (SentiWordNet + Bing Liu w/ Negation)...")
    pos_words = load_wordlist(str(POS_WORDS_PATH))
    neg_words = load_wordlist(str(NEG_WORDS_PATH))
    y_text_all = df["final_label"].astype(str).str.strip().str.capitalize().tolist()

    for scheme_name, texts in schemes.items():
        toks_list = [(t or "").split() for t in texts]
        _, y_test, _, toks_test = train_test_split(
            y_text_all,
            toks_list,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        def _score(preds_text):
            y_true = list(y_test)
            y_pred = [str(v).strip().capitalize() for v in preds_text]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_val = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            return acc, prec, rec, f1_val

        try:
            preds = [sentiwordnet_style_predict(toks, set(), set())[1] for toks in toks_test]
            acc, prec, rec, f1_val = _score(preds)
            add_result("Lexical", f"{scheme_name}/lexical/sentiwordnet", scheme_name, "SentiWordNet", acc, prec, rec, f1_val, np.nan)
        except Exception as e:
            print(f"   [!] SentiWordNet failed for {scheme_name}: {e}")

        preds = [bing_liu_predict_with_negation(toks, pos_words, neg_words)[1] for toks in toks_test]
        acc, prec, rec, f1_val = _score(preds)
        add_result("Lexical", f"{scheme_name}/lexical/bing_liu_negation", scheme_name, "BingLiuNegation", acc, prec, rec, f1_val, np.nan)
else:
    print("\n[WARN] Lexical models skipped (Task 3 lexicon code/files not available).")

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
ml_only = results_df[results_df["Category"] == "Machine Learning"].reset_index(drop=True)
best_model_row = ml_only.iloc[0]
print("🎉 WINNING CONFIGURATION (Deployed Object) 🎉")
print(f"Category: {best_model_row['Category']}")
print(f"Model:    {best_model_row['Model']}")
print(f"Scheme:   {best_model_row['Scheme']}")
print(f"Features: {best_model_row['Features']}")
print(f"Score:    {best_model_row['Accuracy']*100:.2f}% Test Accuracy")
print("="*80)

# Persist the best configuration for downstream error analysis (before/after optimization reporting)
if best_meta:
    DOCS_DIR.mkdir(exist_ok=True)
    with open(DOCS_DIR / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_meta, f, indent=2)

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
