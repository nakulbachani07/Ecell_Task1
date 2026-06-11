import sys
sys.path.append(".")

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import load_sample_dataset
from src.preprocess import combine_text_sections, clean_text, truncate_text
from src.label_creator import create_risk_labels
from src.features import create_tfidf_features
from src.train import train_xgboost

df = load_sample_dataset(split_name="001", sample_size=3000)

print("Original shape:", df.shape)

df["combined_text"] = df.apply(combine_text_sections, axis=1)
df["clean_text"] = df["combined_text"].apply(clean_text)
df["clean_text"] = df["clean_text"].apply(
    lambda x: truncate_text(x, max_chars=10000)
)

df = df[df["clean_text"].str.len() > 500]

print("Shape after removing short rows:", df.shape)

df = create_risk_labels(df, text_column="clean_text")

print("Label distribution:")
print(df["risk_label"].value_counts())

X, vectorizer = create_tfidf_features(df["clean_text"], max_features=5000)
y = df["risk_label"]

print("TF-IDF shape:", X.shape)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nLabel mapping:")
for class_name, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(class_name, "->", encoded_value)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

best_model = train_xgboost(X_train, y_train)

print("Best model training completed.")

os.makedirs("models", exist_ok=True)

# save model, vectorizer , and label encoder
joblib.dump(best_model, "models/xgboost_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
joblib.dump(label_encoder,"models/label_encoder.joblib")

print("Saved files:")
print("models/xgboost_model.joblib")
print("models/tfidf_vectorizer.joblib")
print("models/label_encoder.joblib")

