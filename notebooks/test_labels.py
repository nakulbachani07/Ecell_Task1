import sys
sys.path.append(".")

from src.utils import load_sample_dataset
from src.preprocess import combine_text_sections,clean_text,truncate_text
from src.label_creator import create_risk_labels

#load a small sample safely
df = load_sample_dataset(split_name="001", sample_size=100)

print("original shape:" , df.shape)

#combine important sec into one text
df["combined_text"] = df.apply(combine_text_sections, axis=1)

#clean combined text
df["clean_text"] = df["combined_text"].apply(clean_text)

#truncate long text
df["clean_text"] = df["clean_text"].apply(
    lambda x : truncate_text(x, max_chars=10000)
)

# remove rows with very short text
df = df[df["clean_text"].str.len() > 500]

print("shape after removing short rows", df.shape)

print("\nClean text preview:")
print(df["clean_text"].head(3))

print("Clean text length:")
print(df["clean_text"].str.len().head(10))

#create risk labels
df = create_risk_labels(df, text_column="clean_text")

#print sample
print("\nSample risk scores and labels:")
print(df[["company_name", "risk_score","risk_label"]].head(10))

# check label distribution
print("Risk label distribution:")
print(df["risk_label"].value_counts())

#check risk score stats
print("Risk score stats:")
print(df["risk_score"].describe())