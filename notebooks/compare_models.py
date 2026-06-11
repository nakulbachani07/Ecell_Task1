import sys
sys.path.append(".")

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import load_sample_dataset
from src.preprocess import combine_text_sections, clean_text, truncate_text
from src.label_creator import create_risk_labels
from src.features import create_tfidf_features
from src.train import train_adaboost, train_xgboost, train_catboost
from src.evaluate import evaluate_model

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

print("Label mapping:")
for class_name, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(class_name, "->", encoded_value)

X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

_, _, y_train_text, y_test_text = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed.")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

all_results = []

adaboost_model = train_adaboost(X_train, y_train_text)
adaboost_pred = adaboost_model.predict(X_test)

adaboost_results = evaluate_model(
    y_test=y_test_text,
    y_pred=adaboost_pred,
    model_name="AdaBoost"
)

all_results.append(adaboost_results)

xgboost_model = train_xgboost(X_train, y_train_encoded)
xgboost_pred = xgboost_model.predict(X_test)

xgboost_results = evaluate_model(
    y_test=y_test_encoded,
    y_pred=xgboost_pred,
    model_name="XGBoost",
    target_names=label_encoder.classes_
)

all_results.append(xgboost_results)

catboost_model = train_catboost(X_train, y_train_encoded)
catboost_pred = catboost_model.predict(X_test)
catboost_pred = catboost_pred.flatten()

catboost_results = evaluate_model(
    y_test=y_test_encoded,
    y_pred=catboost_pred,
    model_name="CatBoost",
    target_names=label_encoder.classes_
)

all_results.append(catboost_results)

# final comparison table
comparison_df = pd.DataFrame(all_results)

comparison_df = comparison_df[
    ["model","accuracy","precision","recall","f1_score"]
]

print("Final Model Comparison:")
print(comparison_df)

# Find best model
best_model_row = comparison_df.sort_values(
    by="f1_score",
    ascending=False
).iloc[0]

print("Best model based on F1 score")
print(best_model_row)

