from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(y_test,y_pred,model_name, target_names=None):
    """
    Evaluates a classification model using common metrics
    
    Args:
        y_test: Actual labels
        y_pred: Predicted labels
        model_name: Name of the model being evaluated.
        target_names: Optional list of class names
        
    Returns:
        results: Dictionary containing evaluation metrices
    """

    accuracy = accuracy_score(y_test,y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"{model_name} Evaluation results")
    print("-" * 40)
    print("Accuracy:" , accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0
        ))
    
    print("Confusion matrix:")
    cm = confusion_matrix(y_test,y_pred)
    print(cm)

    results = {
        "model" : model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    return results
    
    # confusion matrix :  high low med
                    #high
                    #low
                    #med
