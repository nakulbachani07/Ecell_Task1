import pandas as pd

RISK_KEYWORDS = [
    "risk",
    "uncertain",
    "uncertainity",
    "loss",
    "losses",
    "debt",
    "litigation",
    "regulation",
    "competition",
    "decline",
    "inflation",
    "cybersecurity",
    "supply chain",
    "material adverse",
    "dafault",
    "recession"
]

def calculate_risk_score(text):
    """
    Calculate a simple risk score by counting risk-related keywords
    """

    if text is None or pd.isna(text):
        return 0
    
    text = str(text).lower()

    score = 0

    for keyword in RISK_KEYWORDS:
        score += text.count(keyword)

    return score

def create_risk_labels(df, text_column="clean_text"):
    """
    Creates Low Risk , Medium Risk , and High Risk labels using risk scores
    calculated from cleaned combined text
    """

    df = df.copy()

    df["risk_score"] = df[text_column].apply(calculate_risk_score)

    low_threshold = df["risk_score"].quantile(0.33)
    high_threshold = df["risk_score"].quantile(0.66)

    def assign_label(score):
        if score <= low_threshold:
            return "Low Risk"
        elif score <= high_threshold:
            return "Medium Risk"
        else:
            return "High Risk"
        
    df["risk_label"] = df["risk_score"].apply(assign_label)

    return df