# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# import joblib
#
# # Load dataset
# df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
# df = df.dropna(subset=['article', 'label'])
# df = df[df['label'].isin(['positive', 'neutral', 'negative'])]
#
# # SBERT Embedding
# sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = sbert_model.encode(df['article'].tolist(), show_progress_bar=True)
#
# # Encode labels
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['label'])
#
# # Balance the dataset
# sm = SMOTE(random_state=42)
# X_resampled, y_resampled = sm.fit_resample(embeddings, y)
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
# )
#
# # Train classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# # Results
# print("\n✅ SBERT + RandomForest Results")
# print(classification_report(y_test, y_pred, zero_division=0))
# print("\n🔍 Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
#
# # Define SBERT wrapper for inference compatibility
# class SBERTTransformer:
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)
#
#     def transform(self, sentences):
#         return self.model.encode(sentences)
#
#     def fit(self, X, y=None):
#         return self
#
# # Save components
# vectorizer = SBERTTransformer()  # Wraps SBERT model
# pipeline = {
#     "vectorizer": vectorizer,
#     "model": clf,
#     "label_encoder": label_encoder
# }
#
# joblib.dump(pipeline, "D:/Python_files/models/sentiment_pipeline.joblib")
# print("✅ Model saved successfully to sentiment_pipeline.joblib")
#
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# import joblib
# import numpy as np
#
# # Load dataset
# df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
# df = df.dropna(subset=['article', 'label'])
# df = df[df['label'].isin(['positive', 'neutral', 'negative'])]
#
# # SBERT Embedding
# sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = sbert_model.encode(df['article'].tolist(), show_progress_bar=True)
#
# # Encode labels
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['label'])
#
# # Balance the dataset
# sm = SMOTE(random_state=42)
# X_resampled, y_resampled = sm.fit_resample(embeddings, y)
#
# # Stratified K-Fold Cross Validation
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# all_reports = []
# fold = 1
#
# for train_index, test_index in kf.split(X_resampled, y_resampled):
#     print(f"\n🔁 Fold {fold}")
#     X_train, X_test = X_resampled[train_index], X_resampled[test_index]
#     y_train, y_test = y_resampled[train_index], y_resampled[test_index]
#
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#
#     report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, output_dict=True)
#     all_reports.append(report)
#
#     print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#     fold += 1
#
# # Average report (macro avg)
# avg_report = {}
# for label in label_encoder.classes_:
#     avg_report[label] = {
#         metric: np.mean([rep[label][metric] for rep in all_reports])
#         for metric in ['precision', 'recall', 'f1-score']
#     }
#
# print("\n📊 Average Classification Report across folds:")
# for label, metrics in avg_report.items():
#     print(f"\nLabel: {label}")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value:.4f}")
#
# # Save final model from last fold (or retrain on full data if preferred)
# final_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# final_clf.fit(X_resampled, y_resampled)
#
# # Define SBERT wrapper
# class SBERTTransformer:
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)
#
#     def transform(self, sentences):
#         return self.model.encode(sentences)
#
#     def fit(self, X, y=None):
#         return self
#
# # Save final pipeline
# vectorizer = SBERTTransformer()
# pipeline = {
#     "vectorizer": vectorizer,
#     "model": final_clf,
#     "label_encoder": label_encoder
# }
#
# joblib.dump(pipeline, "D:/Python_files/models/sentiment_pipeline.joblib")
# print("\n✅ Final model saved successfully to sentiment_pipeline.joblib")


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
from tqdm import tqdm

# --- 1. Data Loading and Preparation ---
print("🔄 Loading and preparing data...")
df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
df = df.dropna(subset=['article', 'label'])
df = df[df['label'].isin(['positive', 'neutral', 'negative'])]
print("✅ Data loaded successfully.")

# --- 2. SBERT Embedding with Chunking and Averaging ---
print("🧠 Initializing SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define chunking parameters
# We use a chunk size smaller than the model's max sequence length (256)
CHUNK_SIZE = 200
OVERLAP = 50

all_article_embeddings = []
print(f"🚀 Generating embeddings with chunking (Chunk size: {CHUNK_SIZE}, Overlap: {OVERLAP})...")

# Use tqdm for a progress bar as this process is slower
for article in tqdm(df['article'].tolist(), desc="Embedding Articles"):
    # Split article into words
    words = article.split()

    # If the article is short, no chunking is needed
    if len(words) <= CHUNK_SIZE:
        article_embedding = sbert_model.encode([article])
    else:
        # Create overlapping chunks
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append(chunk)

        # Encode each chunk and store their embeddings
        chunk_embeddings = sbert_model.encode(chunks)

        # Average the embeddings of all chunks to get a single vector
        article_embedding = np.mean(chunk_embeddings, axis=0, keepdims=True)

    all_article_embeddings.append(article_embedding[0])

# Convert the list of embeddings to a NumPy array
embeddings = np.array(all_article_embeddings)
print("✅ Embeddings generated successfully.")

# --- 3. Encode Labels ---
print("🏷️ Encoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# --- 4. Balance the Dataset ---
print("⚖️ Balancing the dataset with SMOTE...")
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(embeddings, y)
print(f"Dataset balanced. Original samples: {len(y)}, Resampled samples: {len(y_resampled)}")

# --- 5. Stratified K-Fold Cross Validation ---
print("🔄 Starting 5-Fold Cross-Validation...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_reports = []
fold = 1

for train_index, test_index in kf.split(X_resampled, y_resampled):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0,
                                   output_dict=True)
    all_reports.append(report)

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    fold += 1

# --- 6. Average Report Calculation ---
avg_report = {}
for label in label_encoder.classes_:
    avg_report[label] = {
        metric: np.mean([rep[label][metric] for rep in all_reports])
        for metric in ['precision', 'recall', 'f1-score']
    }

print("\n📊 Average Classification Report Across All Folds:")
for label, metrics in avg_report.items():
    print(f"\nLabel: {label}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# --- 7. Final Model Training ---
print("\n💪 Training final model on the full, balanced dataset...")
final_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
final_clf.fit(X_resampled, y_resampled)
print("✅ Final model trained.")


# --- 8. Define SBERT Wrapper with Chunking Logic ---
# This class is CRITICAL for the saved pipeline to work correctly on new, long text.
class SBERTTransformer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = 200
        self.overlap = 50

    def transform(self, sentences):
        """
        Transforms a list of sentences (articles) into embeddings using chunking.
        """
        all_embeddings = []
        for sentence in tqdm(sentences, desc="Vectorizing new data"):
            words = sentence.split()
            if len(words) <= self.chunk_size:
                embedding = self.model.encode([sentence])
            else:
                chunks = []
                for i in range(0, len(words), self.chunk_size - self.overlap):
                    chunk = " ".join(words[i:i + self.chunk_size])
                    chunks.append(chunk)

                chunk_embeddings = self.model.encode(chunks)
                embedding = np.mean(chunk_embeddings, axis=0, keepdims=True)

            all_embeddings.append(embedding[0])
        return np.array(all_embeddings)

    def fit(self, X, y=None):
        # This model is already pre-trained, so fit does nothing.
        return self


# --- 9. Save Final Pipeline ---
print("💾 Saving the final pipeline to disk...")
vectorizer = SBERTTransformer()
pipeline = {
    "vectorizer": vectorizer,
    "model": final_clf,
    "label_encoder": label_encoder
}

joblib.dump(pipeline, "D:/Python_files/models/sentiment_pipeline_chunking.joblib")
print("\n✅ Final model saved successfully to sentiment_pipeline_chunking.joblib")

