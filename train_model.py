# ============================================================
# train_model.py — ML Model Training with TF-IDF + Scikit-learn
# ============================================================
#
# Usage:
#   python train_model.py
#
# Output:
#   models/resume_classifier.pkl   (trained classifier)
#   models/tfidf_vectorizer.pkl    (fitted vectorizer)
#   models/label_encoder.pkl       (label encoder)
#   models/training_report.txt     (classification report)
# ============================================================

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Sklearn imports ─────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

# ─── Paths ───────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "resume_dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH      = os.path.join(MODELS_DIR, "resume_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH    = os.path.join(MODELS_DIR, "label_encoder.pkl")
REPORT_PATH     = os.path.join(MODELS_DIR, "training_report.txt")

# ─── Job Categories ──────────────────────────────────────
CATEGORIES = [
    "Data Scientist",
    "Web Developer",
    "Java Developer",
    "Python Developer",
    "HR",
    "AI Engineer",
]

# ─── Text Cleaning ───────────────────────────────────────
import re

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"[^a-z0-9\s\+#]", " ", text)          # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Data Loading ────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"[INFO] Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Category", "Resume"])
    df["Category"] = df["Category"].str.strip()
    df["Resume"]   = df["Resume"].apply(clean_text)
    print(f"[INFO] Loaded {len(df)} samples across {df['Category'].nunique()} categories")
    print(df["Category"].value_counts().to_string())
    return df

# ─── Data Augmentation (simple synonym replacement) ──────
def augment_data(df: pd.DataFrame, factor: int = 3) -> pd.DataFrame:
    """
    Simple augmentation by appending common synonyms/related terms
    to increase training size.
    """
    augmentations = {
        "Data Scientist":  ["data analysis", "statistical modeling", "predictive analytics", "big data"],
        "Web Developer":   ["frontend", "backend", "fullstack", "web application", "responsive design"],
        "Java Developer":  ["object oriented", "enterprise java", "spring framework", "jvm"],
        "Python Developer":["python scripting", "automation", "backend development", "api development"],
        "HR":              ["human resources", "people management", "recruitment", "talent management"],
        "AI Engineer":     ["artificial intelligence", "neural network", "llm", "ai model", "generative ai"],
    }
    rows = []
    for _, row in df.iterrows():
        extra = " ".join(augmentations.get(row["Category"], []))
        for _ in range(factor - 1):
            rows.append({"Category": row["Category"], "Resume": row["Resume"] + " " + extra})
    augmented = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    augmented = augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    return augmented

# ─── Model Training ──────────────────────────────────────
def train_best_model(X_train, X_test, y_train, y_test):
    """Train multiple models, pick the best one by CV accuracy."""
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5, solver="lbfgs", multi_class="auto"),
        "Linear SVC":          LinearSVC(C=1.0, max_iter=2000),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        stop_words="english",
        min_df=1,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    best_model = None
    best_score = 0
    best_name  = ""
    results    = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[INFO] Evaluating candidate models …")
    print("-" * 55)
    for name, clf in candidates.items():
        try:
            cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=cv, scoring="accuracy")
            mean_cv   = cv_scores.mean()
            results[name] = {"cv_mean": mean_cv, "cv_std": cv_scores.std()}
            print(f"  {name:<25}  CV Acc: {mean_cv:.4f} ± {cv_scores.std():.4f}")
            if mean_cv > best_score:
                best_score = mean_cv
                best_model = clf
                best_name  = name
        except Exception as e:
            print(f"  {name:<25}  FAILED: {e}")

    print("-" * 55)
    print(f"\n[INFO] Best model: {best_name} (CV Acc={best_score:.4f})")

    # Final fit with full training data
    best_model.fit(X_train_vec, y_train)
    y_pred = best_model.predict(X_test_vec)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    print(f"\n[RESULT] Test Accuracy : {acc:.4f}")
    print(f"[RESULT] Weighted F1   : {f1:.4f}")
    print("\n[RESULT] Classification Report:")
    print(report)

    return best_model, vectorizer, results, report, acc, f1, best_name


# ─── Save Artifacts ──────────────────────────────────────
def save_artifacts(model, vectorizer, encoder, report_text):
    with open(MODEL_PATH,      "wb") as f: pickle.dump(model,      f)
    with open(VECTORIZER_PATH, "wb") as f: pickle.dump(vectorizer, f)
    with open(ENCODER_PATH,    "wb") as f: pickle.dump(encoder,    f)
    with open(REPORT_PATH,     "w")  as f: f.write(report_text)
    print(f"\n[SAVED] Model      -> {MODEL_PATH}")
    print(f"[SAVED] Vectorizer -> {VECTORIZER_PATH}")
    print(f"[SAVED] Encoder    -> {ENCODER_PATH}")
    print(f"[SAVED] Report     -> {REPORT_PATH}")


# ─── Prediction Utility ──────────────────────────────────
def load_model():
    """Load trained artifacts. Returns (model, vectorizer, encoder)."""
    with open(MODEL_PATH,      "rb") as f: model      = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
    with open(ENCODER_PATH,    "rb") as f: encoder    = pickle.load(f)
    return model, vectorizer, encoder


def predict_category(text: str, model, vectorizer, encoder) -> dict:
    """
    Predict the job category for a given resume text.
    Returns: { 'category': str, 'confidence': float, 'all_scores': dict }
    """
    cleaned  = clean_text(text)
    vec      = vectorizer.transform([cleaned])
    pred_idx = model.predict(vec)[0]

    # Confidence (probability or decision function)
    all_scores = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        for i, cls in enumerate(model.classes_):
            label = encoder.inverse_transform([cls])[0] if hasattr(encoder, "inverse_transform") else str(cls)
            all_scores[label] = round(float(probs[i]) * 100, 1)
        confidence = float(probs.max()) * 100
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        # Normalize to [0,1] softmax-style
        exp_s = np.exp(scores - scores.max())
        probs = exp_s / exp_s.sum()
        for i, cls in enumerate(model.classes_):
            label = encoder.inverse_transform([cls])[0] if hasattr(encoder, "inverse_transform") else str(cls)
            all_scores[label] = round(float(probs[i]) * 100, 1)
        confidence = float(probs.max()) * 100
    else:
        confidence = 0.0

    if hasattr(encoder, "inverse_transform"):
        category = encoder.inverse_transform([pred_idx])[0]
    else:
        category = str(pred_idx)

    return {
        "category":   category,
        "confidence": round(confidence, 1),
        "all_scores": all_scores,
    }


# ─── Main ────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  AI Resume Screener — Model Training")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # 1. Load data
    df = load_data()

    # 2. Encode labels
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["Category"])
    print(f"\n[INFO] Classes: {list(encoder.classes_)}")

    # 3. Augment
    df = augment_data(df, factor=4)
    df = df.dropna(subset=["label", "Resume"])
    df["label"] = df["label"].astype(int)
    print(f"[INFO] After augmentation: {len(df)} samples")

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["Resume"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 5. Train
    model, vectorizer, cv_results, report, acc, f1, best_name = train_best_model(
        X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()
    )

    # 6. Build report text
    report_text = (
        f"Training Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*55}\n"
        f"Best Model : {best_name}\n"
        f"Test Acc   : {acc:.4f}\n"
        f"Weighted F1: {f1:.4f}\n\n"
        f"Cross-Validation Results:\n"
        + "\n".join(f"  {k}: {v['cv_mean']:.4f} ± {v['cv_std']:.4f}" for k, v in cv_results.items())
        + f"\n\nClassification Report:\n{report}"
    )

    # 7. Save
    save_artifacts(model, vectorizer, encoder, report_text)

    print("\n[DONE] Training complete! Run `streamlit run app.py` to start the tool.")


if __name__ == "__main__":
    main()
