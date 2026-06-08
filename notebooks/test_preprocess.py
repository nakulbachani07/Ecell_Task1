# This file is a testing script. Its job is to check whether your utils.py and preprocess.py are working together properly.
import sys #sys is a Python module that lets you interact with Python’s system settings
sys.path.append(".")
#tells Python: Also search for imports from the current project folder.

from src.utils import load_sample_dataset
from src.preprocess import combine_text_sections, clean_text, truncate_text

df = load_sample_dataset(sample_size=10, split_name="001")

print("original shape:", df.shape)
print("original columns:", df.columns.to_list())

df["combined_text"] = df.apply(combine_text_sections,axis=1) # this line creates new column called combined_text to every row , axis=0 → apply column-wise, axis=1 → apply row-wise 
df["clean_text"] = df["combined_text"].apply(clean_text)
df["clean_text"] = df["clean_text"].apply(
    lambda x: truncate_text(x , max_chars=10000)
)

df = df[df["clean_text"].str.len()>500]

print("\ncleaned text:")
print(df["clean_text"].iloc[0][:1000])
# iloc[0][:1000] - This prints the first 1000 characters of the cleaned text from the first row.

print("\nAfter removing very short rows:")
print("New shape:", df.shape)

print("\nCleaned text length:")
print(df["clean_text"].str.len())

print("\nAverage cleaned text length:")
print(df["clean_text"].str.len().mean())
