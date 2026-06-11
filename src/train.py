from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def train_adaboost(X_train, y_train):
    """
    Train an AdaBoost classifier.
    
    Args:
       X_train: Training features created using TF-IDF
       y_train: Training labels
       
    Returns:
       model: Trained AdaBoost model
    """

    base_model = DecisionTreeClassifier( # weak model used inside AdaBoost
        max_depth=1,
        random_state=42
    )

    model = AdaBoostClassifier(
        estimator=base_model,
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )

    model.fit(X_train,y_train)

    return model


def train_xgboost(X_train,y_train):
    """
    Trains an XGBoost classifier.
    
    Args:
        X_train: Training features created using TF-IDF.
        y_train: Encoded training labels
        
    Returns:
        model: Trained XGBoost model.
    """

    model = XGBClassifier(
        n_estimators=100, #Use 100 boosting rounds / 100 trees.
        learning_rate = 0.1,
        max_depth=4, #Each tree can make decisions up to depth 4.
        objective="multi:softprob", #This is a multi-class classification problem.,Return probabilities for each class.
        eval_metric="mlogloss", #It measures how confident and correct the model's probability predictions are.
        random_state=42
    )

    model.fit(X_train,y_train)

    return model

#It combines many decision trees to create one strong model.
def train_catboost(X_train,y_train):
    """
    Trains a CatBoost classifier.

    Args:
        X_train: Training features created using TF-IDF.
        y_train: Encoded training labels.

    Returns:
        model: Trained CatBoost model.
    """

    model = CatBoostClassifier(
        iterations=100,#Build 100 boosting trees/rounds.
        learning_rate=0.1,
        depth=4,
        loss_function="MultiClass", #This is a multi-class classification problem.
        random_seed=42,
        verbose=False #Prevents CatBoost from printing training logs for every iteration.
    )

    model.fit(X_train,y_train)

    return model