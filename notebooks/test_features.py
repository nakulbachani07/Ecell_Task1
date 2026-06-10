import sys
sys.path.append(".")

from sklearn.model_selection import train_test_split

from src.utils import load_sample_dataset
from src.preprocess import combine_text_sections,clean_text,truncate_text
from src.label_creator import create_risk_labels
from src.features import create_tfidf_features

#Load dataset safely
df = load_sample_dataset(split_name="001", sample_size=100)

print("Original shape:", df.shape)


#  Combine text sections
df["combined_text"] = df.apply(combine_text_sections, axis=1)


# Clean text
df["clean_text"] = df["combined_text"].apply(clean_text)


# Truncate text
df["clean_text"] = df["clean_text"].apply(
    lambda x: truncate_text(x, max_chars=10000)
)


# Remove very short rows
df = df[df["clean_text"].str.len() > 500]

print("Shape after removing short rows:", df.shape)

#Create labels
df = create_risk_labels(df, text_column="clean_text")

print("Label distribution:")
print(df["risk_label"].value_counts())

#Create X and y
X, vectorizer = create_tfidf_features(df["clean_text"], max_features=5000)
y = df["risk_label"]

print("TF_IDF feature matrix shape:", X.shape)
print("Number of labels:", y.shape)

#Train - test split
X_train , X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,#20% for test , and 80% for training
    random_state=42, #with this , result stays consistent.
    stratify=y # keeps class distribution balanced in train test split
    
)

print("Train test split completed")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#show some vocabulary words
features_names = vectorizer.get_feature_names_out()

print("Sample TF-IDF features:")
print(features_names[:30])