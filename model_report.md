## Dataset Handling

The dataset is large, so I avoided loading the full dataset directly. I used the Hugging Face `datasets` library with `streaming=True` and loaded only a controlled sample from split `001`.

## Dataset Structure

After inspection, I found that the dataset is divided into numbered splits such as `001`, `002`, etc., rather than a standard `train` split. Each row contains one company filing with columns such as `Business`, `Risk Factors`, `Management’s Discussion and Analysis`, and `Financial Statements`.

## Preprocessing

For initial preprocessing, I selected the `Business`, `Risk Factors`, and `Management’s Discussion and Analysis` sections. These sections were combined into a single text field, cleaned using regular expressions, lowercased, normalized for whitespace, and truncated to 10,000 characters to control memory and training cost.

Rows with very short cleaned text were removed because they did not contain enough information for reliable classification.
