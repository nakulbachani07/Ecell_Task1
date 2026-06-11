import sys
sys.path.append(".")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.utils import load_sample_dataset
from src.preprocess import combine_text_sections, clean_text, truncate_text
from src.label_creator import create_risk_labels
from src.features import create_tfidf_features
from src.train import train_adaboost

df = load_sample_dataset(split_name="001", sample_size=300)
print("original shape" , df.shape)

df["combined_text"] = df.apply(combine_text_sections, axis=1)

df["clean_text"] = df["combined_text"].apply(clean_text)

df["clean_text"] = df["clean_text"].apply(
    lambda x: truncate_text(x, max_chars=10000)
)

df = df[df["clean_text"].str.len() > 500]

print("shape after removing short rows:" , df.shape)

df = create_risk_labels(df, text_column="clean_text")

print("Label distribution:")
print(df["risk_label"].value_counts())

X, vectorizer = create_tfidf_features(df["clean_text"], max_features=5000)
y = df["risk_label"]

print("TF-IDF shape:", X.shape)

X_train , X_test, y_train , y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split completed")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

model = train_adaboost(X_train,y_train)

print("AdaBoost training completed.")

#make predictions
y_pred = model.predict(X_test)

#evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("AdaBoost Accuracy:" , accuracy)

print("Classification report:")
print(classification_report(y_test,y_pred))


