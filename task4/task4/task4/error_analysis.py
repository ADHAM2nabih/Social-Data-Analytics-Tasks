import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import json

warnings.filterwarnings('ignore')

print("Starting Error Analysis (Task 4)...")

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


def load_best_config() -> dict:
    cfg_path = DOCS_DIR / "best_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {"scheme": "v1_basic", "features": "Bag-Of-Words", "model_key": "Multinomial Naive Bayes", "uses_pca": False, "test_size": 0.2, "random_state": 42}


def make_baseline_model(model_key: str):
    if model_key.startswith("Multinomial Naive Bayes"):
        return MultinomialNB()
    if model_key.startswith("Gaussian Naive Bayes"):
        return GaussianNB()
    if model_key.startswith("Decision Tree"):
        return DecisionTreeClassifier(random_state=42)
    if model_key.startswith("Random Forest"):
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    return MultinomialNB()

try:
    classifier = joblib.load(MODELS_DIR / "best_classifier_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    
    pca_path = MODELS_DIR / "pca_transformer.pkl"
    if pca_path.exists():
        pca = joblib.load(pca_path)
    else:
        pca = None
except FileNotFoundError:
    print("[ERROR] Required pickle files not found. Run 'best_model_selection.py' first.")
    exit()

# Load Text and Features
data_path = DATA_PATH
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"[!] Critical Error: Cannot find labeled data at {data_path}")
    exit(1)

for col in ['v1_basic_text', 'v2_no_stop_text', 'v3_stem_text']:
    if col in df.columns:
        df[col] = df[col].fillna("")
        
df['final_label'] = df['final_label'].astype(str).str.strip().str.capitalize()

reviews = df["original_selftext"].fillna(df["original_title"]).fillna("").values

cfg = load_best_config()
scheme = str(cfg.get("scheme", "v1_basic"))
text_col = f"{scheme}_text"
if text_col not in df.columns:
    text_col = "v1_basic_text"
texts = df[text_col].values

y_raw = df['final_label'].values
y = le.transform(y_raw)

# Vectorization via loaded transformer
X_bow = vectorizer.transform(texts).toarray()

# Split identically to training
indices = np.arange(len(y))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_bow, y, indices, test_size=0.2, random_state=42, stratify=y
)

if pca is not None:
    X_test = pca.transform(X_test)

# ----------------------------
# BEFORE vs AFTER optimization (best model only)
# ----------------------------
baseline_model = make_baseline_model(str(cfg.get("model_key", "Multinomial Naive Bayes")))
if pca is not None:
    X_train_fit = pca.transform(X_train)
    X_test_fit = X_test
else:
    X_train_fit = X_train
    X_test_fit = X_test

baseline_model.fit(X_train_fit, y_train)
y_pred_before = baseline_model.predict(X_test_fit)

# Predict with optimized (loaded) classifier
y_pred_after = classifier.predict(X_test_fit)

acc_before = accuracy_score(y_test, y_pred_before)
f1_before = f1_score(y_test, y_pred_before, average='weighted', zero_division=0)
acc_after = accuracy_score(y_test, y_pred_after)
f1_after = f1_score(y_test, y_pred_after, average='weighted', zero_division=0)

print("\n" + "="*80)
print("📊 BEST MODEL: BEFORE vs AFTER OPTIMIZATION")
print("="*80)
print(f"Scheme: {scheme} | Features: {cfg.get('features')} | Model: {cfg.get('model_key')}")
print(f"BEFORE  -> Accuracy: {acc_before:.3f} | F1 (weighted): {f1_before:.3f}")
print(f"AFTER   -> Accuracy: {acc_after:.3f} | F1 (weighted): {f1_after:.3f}")

report_before = classification_report(y_test, y_pred_before, target_names=le.classes_, zero_division=0)
report_after = classification_report(y_test, y_pred_after, target_names=le.classes_, zero_division=0)
DOCS_DIR.mkdir(exist_ok=True)
(DOCS_DIR / "classification_report_before.txt").write_text(report_before, encoding="utf-8")
(DOCS_DIR / "classification_report_after.txt").write_text(report_after, encoding="utf-8")

# Random chance baseline (stratified) to report expected-ish F1/accuracy
dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(X_train_fit, y_train)
y_pred_dummy = dummy.predict(X_test_fit)
acc_dummy = accuracy_score(y_test, y_pred_dummy)
f1_dummy = f1_score(y_test, y_pred_dummy, average='weighted', zero_division=0)
print(f"RANDOM  -> Accuracy: {acc_dummy:.3f} | F1 (weighted): {f1_dummy:.3f} (DummyClassifier stratified)")

(DOCS_DIR / "best_model_metrics.json").write_text(
    json.dumps(
        {
            "scheme": scheme,
            "text_col": text_col,
            "model_key": cfg.get("model_key"),
            "features": cfg.get("features"),
            "before": {"accuracy": acc_before, "f1_weighted": f1_before},
            "after": {"accuracy": acc_after, "f1_weighted": f1_after},
            "random_chance": {"accuracy": acc_dummy, "f1_weighted": f1_dummy, "strategy": "stratified"},
        },
        indent=2,
    ),
    encoding="utf-8",
)

# Confusion Matrices
cm_before = confusion_matrix(y_test, y_pred_before)
cm_after = confusion_matrix(y_test, y_pred_after)
print(f"\n[INFO] Confusion Matrix mapping: {list(le.classes_)}")
print("[BEFORE]\n", cm_before)
print("[AFTER ]\n", cm_after)

# Save visual confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Best Model (Before Optimization)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
DOCS_DIR.mkdir(exist_ok=True)
before_out = DOCS_DIR / "confusion_matrix_before.png"
plt.savefig(before_out)

plt.figure(figsize=(8,6))
sns.heatmap(cm_after, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Best Model (After Optimization)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
after_out = DOCS_DIR / "confusion_matrix_after.png"
plt.savefig(after_out)

# Keep existing filename for the frontend
confusion_out = DOCS_DIR / "confusion_matrix.png"
plt.savefig(confusion_out)
print(f"[INFO] Saved confusion matrices to {before_out} and {after_out}")

# Find misclassifications
errors = []
for i, (true_idx, pred_idx, orig_idx) in enumerate(zip(y_test, y_pred_after, idx_test)):
    if true_idx != pred_idx:
        errors.append({
            "Review": reviews[orig_idx],
            "True_Label": le.classes_[true_idx],
            "Predicted_Label": le.classes_[pred_idx]
        })

err_df = pd.DataFrame(errors)
DOCS_DIR.mkdir(exist_ok=True)
errors_out = DOCS_DIR / "model_errors.csv"
err_df.to_csv(errors_out, index=False)

print(f"\nTotal misclassifications: {len(err_df)} out of {len(y_test)}")
print("-" * 80)
# Show only up to 5 for discussion snippet
for idx, row in err_df.head(5).iterrows():
    print(f"[{idx+1}] TRUE: {row['True_Label']:<8} | PRED: {row['Predicted_Label']:<8}")
    print(f"    TEXT: {str(row['Review']).strip()[:150]}...")
    print("-" * 80)
print(f"\n[INFO] Full errors saved to '{errors_out}' for discussion.")