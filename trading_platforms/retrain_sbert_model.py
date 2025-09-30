# retrain_sbert_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from tqdm import tqdm

# --- 1. IMPORT FROM YOUR NEW PROJECT STRUCTURE ---
import config
from systems.common_classes import SBERTTransformer # Use the centralized class

print("--- Starting SBERT Model Training & Saving Process (with SMOTE) ---")

# --- 2. LOAD DATA USING PORTABLE PATHS ---
try:
    data_path = os.path.join(config.BASE_DIR, 'data', 'fully_merged.csv')
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['article', 'label'])
    df = df[df['label'].isin(['positive', 'neutral', 'negative'])]
    print(f"✅ Data loaded successfully from {data_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find training data at {data_path}.")
    exit()

# --- 3. GENERATE EMBEDDINGS ---
# We still need to generate embeddings once for the SMOTE process.
vectorizer = SBERTTransformer()
print("🚀 Generating embeddings for the entire dataset...")
embeddings = vectorizer.transform(df['article'].tolist())
print("✅ Embeddings generated successfully.")

# --- 4. ENCODE LABELS AND BALANCE DATA ---
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

print("⚖️ Balancing the dataset with SMOTE...")
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(embeddings, y)
print(f"Dataset balanced. Original samples: {len(y)}, Resampled samples: {len(y_resampled)}")

# --- 5. CROSS-VALIDATE THE CLASSIFIER ---
print("🔄 Starting 5-Fold Cross-Validation...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(X_resampled, y_resampled), 1):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# --- 6. TRAIN FINAL MODEL ---
print("\n💪 Training final model on the full, balanced dataset...")
final_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
final_clf.fit(X_resampled, y_resampled)
print("✅ Final model trained.")

# --- 7. SAVE THE FINAL PIPELINE PACKAGE ---
print("💾 Saving the final pipeline to disk...")
# We save the vectorizer instance separately from the trained classifier
final_model_package = {
    "vectorizer": vectorizer,
    "model": final_clf,
    "label_encoder": label_encoder
}

joblib.dump(final_model_package, config.SBERT_MODEL_PATH)
print(f"\n✅ Final model saved successfully to: {config.SBERT_MODEL_PATH}")