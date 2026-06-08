from datasets import load_dataset
import pandas as pd

DATASET_NAME = "winterForestStump/10-K_sec_filings"

def load_sample_dataset(split_name="001", sample_size=500):
    dataset=load_dataset(
        DATASET_NAME,
        split=split_name,
        streaming=True
    )

    rows = []

    for i,row in enumerate(dataset):
        if i>=sample_size:
            break
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__": # run the code below only when this file is run directly ( if we import src.utils to another file , this part of code wont run)
    df = load_sample_dataset(split_name="001", sample_size=5)

    print("Dataset loaded safely.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df)