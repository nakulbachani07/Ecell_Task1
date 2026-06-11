import sys
sys.path.append(".")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from src.utils import load_sample_dataset
from src.preprocess import combine_text_sections, clean_text, truncate_text
from src.label_creator import create_risk_labels
from src.features import create_tfidf_features
from src.train import train_xgboost

df = load_sample_dataset(split_name="001", sample_size=300)

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

#encode labels for XgBoost
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label mapping:")
for class_name, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(class_name, "->" , encoded_value)

#train , test split
X_train , X_test, y_train , y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train Test split completed")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("Unique y_train values:")
print(set(y_train))

model = train_xgboost(X_train,y_train)

print("XGBoost training completed")

#make predictions
y_preds = model.predict(X_test)

#evaluate model
accuracy = accuracy_score(y_test,y_preds)
print("XGBoost Accuracy:", accuracy)

print("Classification Report:")

print(classification_report(y_test,y_preds,target_names=label_encoder.classes_)) # with target_names = report shows readable names
                    

