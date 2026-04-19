import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings

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

# The best model was selected previously based on dynamic schema but to calculate error matrix
# We will use v1_basic as the baseline for testing outputs
texts = df["v1_basic_text"].values

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

# Predict
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n[INFO] Confusion Matrix mapping: {list(le.classes_)}")
print(cm)

# Save visual confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Best Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
DOCS_DIR.mkdir(exist_ok=True)
confusion_out = DOCS_DIR / "confusion_matrix.png"
plt.savefig(confusion_out)
print(f"[INFO] Saved visual confusion matrix to {confusion_out}")

# Find misclassifications
errors = []
for i, (true_idx, pred_idx, orig_idx) in enumerate(zip(y_test, y_pred, idx_test)):
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