import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
df = df.dropna(subset=['article', 'label'])
df = df[df['label'].isin(['positive', 'neutral', 'negative'])]

# TF-IDF Vectorization
X = df['article'].values
y = df['label'].values

vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# SMOTE Oversampling
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_vec, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("\n✅ Balanced TF-IDF + RandomForestClassifier")
print(classification_report(y_test, y_pred, zero_division=0))
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

