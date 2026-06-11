# SEC Filing Risk Classification using Boosting Models

This project is made for the E-Cell AI and Automation Task 1. The aim of this project is to build a basic document intelligence system which can read SEC 10-K filing text and classify the filing into a risk category.

The project uses text data from company 10-K filings and predicts whether the filing belongs to one of these classes:

* Low Risk
* Medium Risk
* High Risk

I used text preprocessing, TF-IDF feature extraction, three boosting models, and finally deployed the best model using FastAPI.

## Dataset Used

Dataset: `winterForestStump/10-K_sec_filings`

The dataset was loaded from Hugging Face using streaming mode. I used streaming because the complete dataset is very large and loading everything at once would take a lot of time and memory. For this task, I loaded a sample of 3000 filings from split `001`.

After cleaning and filtering very short documents, 2054 usable filings were left for training and evaluation.

## Project Workflow

The project was completed in these main steps:

1. Load SEC 10-K filing data using Hugging Face streaming.
2. Combine important filing sections such as Business, Risk Factors and MD&A.
3. Clean the text by removing unnecessary symbols, extra spaces and noisy formatting.
4. Create risk labels using a keyword-based risk scoring method.
5. Convert cleaned text into numerical features using TF-IDF.
6. Train three boosting models:

   * AdaBoost
   * XGBoost
   * CatBoost
7. Evaluate all models using accuracy, precision, recall, F1 score and confusion matrix.
8. Select the best model based on weighted F1 score.
9. Save the final model, vectorizer and label encoder.
10. Deploy the model using FastAPI.

## Folder Structure

```text
Ecelltask_1/
│
├── api/
│   └── app.py
│
├── models/
│   ├── xgboost_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── label_encoder.joblib
│
├── notebooks/
│   ├── compare_models.py
│   ├── save_best_models.py
│   ├── test_features.py
│   ├── test_labels.py
│   ├── test_preprocess.py
│   ├── test_train_adaboost.py
│   ├── train_test_xgboost.py
│   └── train_test_catboost.py
│
├── src/
│   ├── utils.py
│   ├── preprocess.py
│   ├── label_creator.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
├── model_report.md
├── requirements.txt
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/nakulbachani07/Ecell_Task1.git
cd Ecell_Task1
```

### 2. Create and activate virtual environment

```bash
python -m venv Ecelltask_1
source Ecelltask_1/bin/activate
```

For Windows:

```bash
Ecelltask_1\Scripts\activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

## Running the Pipeline

To test data loading:

```bash
python src/utils.py
```

To compare all models:

```bash
python notebooks/compare_models.py
```

To train and save the best model:

```bash
python notebooks/save_best_models.py
```

## Model Performance

The final comparison of the models is shown below:

| Model    | Accuracy | Precision | Recall | F1 Score |
| -------- | -------: | --------: | -----: | -------: |
| AdaBoost |   0.6618 |    0.7057 | 0.6618 |   0.6731 |
| XGBoost  |   0.8127 |    0.8074 | 0.8127 |   0.8089 |
| CatBoost |   0.7859 |    0.7770 | 0.7859 |   0.7732 |

XGBoost performed the best overall, so it was selected as the final model for deployment.

## Running the FastAPI App

Start the API server using:

```bash
uvicorn api.app:app --reload
```

After running the command, open this URL in the browser:

```text
http://127.0.0.1:8000/docs
```

This opens the Swagger UI where the `/predict` endpoint can be tested.

## API Endpoint

### POST `/predict`

Example input:

```json
{
  "text": "The company faces debt, litigation, regulatory pressure, market uncertainty and possible losses due to adverse economic conditions."
}
```

Example output:

```json
{
  "label": "High Risk",
  "confidence": 0.82
}
```

The confidence value can change depending on the input text.

## Important Files

* `src/utils.py` - safely loads the dataset using streaming
* `src/preprocess.py` - combines and cleans filing text
* `src/label_creator.py` - creates Low, Medium and High Risk labels
* `src/features.py` - creates TF-IDF features
* `src/train.py` - trains AdaBoost, XGBoost and CatBoost models
* `src/evaluate.py` - evaluates the models
* `notebooks/compare_models.py` - compares all three models
* `notebooks/save_best_models.py` - trains and saves the best model
* `api/app.py` - FastAPI deployment file

## Final Model

The final selected model is XGBoost because it gave the highest weighted F1 score among the three boosting models.

Saved model files:

```text
models/xgboost_model.joblib
models/tfidf_vectorizer.joblib
models/label_encoder.joblib
```

## Limitations

This project uses a keyword-based method to create labels because the dataset did not already contain direct risk labels. So the labels are useful for this task, but they are not the same as expert human-labelled financial risk categories.

Nakul Bachani
