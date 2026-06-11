# Model Report

## 1. Project Overview

The aim of this project is to classify SEC 10-K filings into financial risk categories. The final output of the system is one of three labels:

* Low Risk
* Medium Risk
* High Risk

This project follows a simple machine learning pipeline. First, the SEC filing text is loaded and cleaned. Then labels are created using a risk scoring method. After that, TF-IDF is used for feature extraction and three boosting models are trained and compared.

The three models used are:

* AdaBoost
* XGBoost
* CatBoost

The best model is then saved and deployed using FastAPI.

## 2. Dataset

The dataset used in this project is:

```text
winterForestStump/10-K_sec_filings
```

This dataset contains SEC 10-K filings of companies. The filings are divided into different sections such as Business, Risk Factors, Management Discussion and Analysis, Financial Statements and other sections.

The complete dataset is large, so I used Hugging Face streaming mode instead of downloading the full dataset at once. I loaded a sample of 3000 filings from split `001`.

After preprocessing and filtering short or empty documents, 2054 filings were used for model training and testing.

## 3. Data Loading Approach

The dataset was loaded using the Hugging Face `datasets` library with `streaming=True`.

This was done because the dataset is very large, and loading the full dataset can be slow and memory-heavy. Streaming allowed me to read only the required number of rows without downloading everything.

The loading function was written in `src/utils.py`.

## 4. Text Sections Used

For each filing, I combined important text sections to create one main text input.

The main sections used were:

* Business
* Risk Factors
* Management’s Discussion and Analysis of Financial Condition and Results of Operations

These sections were selected because they contain useful information about the company, its financial condition, business situation, and possible risks.

## 5. Text Preprocessing

The preprocessing was done in `src/preprocess.py`.

The main preprocessing steps were:

1. Convert text to lowercase.
2. Remove HTML-like tags.
3. Remove URLs.
4. Remove unnecessary special characters.
5. Remove extra spaces.
6. Truncate very long text to control memory and training time.

This helped make the text cleaner and easier for the machine learning models to process.

## 6. Label Creation

The dataset did not directly provide Low Risk, Medium Risk and High Risk labels. So I created labels using a keyword-based risk scoring method.

A list of risk-related words was used, such as:

```text
risk, uncertainty, loss, debt, litigation, regulation, competition, inflation, adverse, default, recession, liability, bankruptcy
```

For each filing, the number of risk-related terms was counted. This count was used as the risk score.

Then the filings were divided into three groups using percentile thresholds:

* Lower scores were assigned Low Risk
* Middle scores were assigned Medium Risk
* Higher scores were assigned High Risk

The label creation logic was written in `src/label_creator.py`.

The final label distribution was:

| Label       | Count |
| ----------- | ----: |
| Low Risk    |   807 |
| High Risk   |   671 |
| Medium Risk |   576 |

## 7. Feature Engineering

TF-IDF vectorization was used for feature extraction.

TF-IDF was selected because it is simple, effective, and suitable for classical machine learning models. Since boosting models cannot directly understand raw text, TF-IDF converts the filing text into numerical features.

The TF-IDF settings used were:

* Maximum features: 5000
* Stop words: English
* N-grams: Unigrams and bigrams
* Numeric-only tokens removed

Using unigrams and bigrams helped the model understand both individual words and short phrases. For example, phrases like “material adverse”, “financial condition” and “market risk” can be useful in SEC filing classification.

Feature engineering was written in `src/features.py`.

The final TF-IDF feature matrix shape was:

```text
(2054, 5000)
```

## 8. Train-Test Split

The data was split into training and testing sets using an 80-20 split.

Stratified splitting was used so that the class distribution stayed balanced in both training and testing data.

Final split:

```text
Training data: 1643 samples
Testing data: 411 samples
```

## 9. Models Trained

### AdaBoost

AdaBoost was trained using a decision tree base estimator. It works by training weak learners one by one and giving more focus to the wrongly classified samples.

### XGBoost

XGBoost is an optimized gradient boosting model. It usually performs well on structured numerical features, and in this project it worked well with TF-IDF features.

### CatBoost

CatBoost is another boosting algorithm. It is commonly useful for categorical data, but it can also be used for classification problems with numerical features.

All three models were trained and compared because the task required using AdaBoost, XGBoost and CatBoost.

## 10. Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Weighted average was used for precision, recall and F1 score because the classes were not perfectly equal in size.

## 11. Model Results

| Model    | Accuracy | Precision | Recall | F1 Score |
| -------- | -------: | --------: | -----: | -------: |
| AdaBoost |   0.6618 |    0.7057 | 0.6618 |   0.6731 |
| XGBoost  |   0.8127 |    0.8074 | 0.8127 |   0.8089 |
| CatBoost |   0.7859 |    0.7770 | 0.7859 |   0.7732 |

## 12. Confusion Matrices

### AdaBoost

```text
[[100   8  26]
 [  0  96  66]
 [ 15  24  76]]
```

### XGBoost

```text
[[114   1  19]
 [  0 150  12]
 [ 27  18  70]]
```

### CatBoost

```text
[[111   5  18]
 [  0 156   6]
 [ 26  33  56]]
```

## 13. Best Model Selection

XGBoost was selected as the final model because it achieved the best overall performance.

It had the highest accuracy and weighted F1 score:

```text
Accuracy: 0.8127
Weighted F1 Score: 0.8089
```

The F1 score was considered important because it balances both precision and recall. Since this is a classification task with three classes, weighted F1 score gives a better overall idea of model performance than accuracy alone.

## 14. API Deployment

The best model was saved using `joblib`.

The following files were saved:

```text
models/xgboost_model.joblib
models/tfidf_vectorizer.joblib
models/label_encoder.joblib
```

A FastAPI app was created in:

```text
api/app.py
```

The API has a `/predict` endpoint which accepts text input and returns the predicted risk label and confidence score.

Example input:

```json
{
  "text": "The company faces debt, litigation, regulatory pressure and possible losses due to market uncertainty."
}
```

Example output:

```json
{
  "label": "High Risk",
  "confidence": 0.82
}
```

## 15. Limitations

The main limitation of this project is that the labels were created using a keyword-based scoring method. This means the model is learning from automatically generated labels, not expert-labelled financial risk data.

Another limitation is that TF-IDF does not deeply understand context. For example, it may treat words based on frequency and importance, but it may not fully understand the meaning of long financial statements.

Also, the model was trained on a sample of the dataset instead of the full dataset, mainly to keep training time and memory usage manageable.

## 16. Future Improvements

Some possible improvements are:

1. Use a larger sample of the dataset.
2. Improve label creation using better financial rules.
3. Use sentence embeddings or transformer models.
4. Try finance-specific language models.
5. Add a better frontend for testing the API.
6. Deploy the API online for a live demo.

## 17. Conclusion

In this project, I built a full text classification pipeline for SEC 10-K filings. The system loads data safely using streaming, preprocesses the filing text, creates risk labels, extracts TF-IDF features, trains three boosting models and deploys the best model using FastAPI.

Among AdaBoost, XGBoost and CatBoost, XGBoost gave the best results and was selected as the final model.

