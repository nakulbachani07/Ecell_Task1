#So features.py will convert text into numbers.
# TF - IDF -> term frequency - inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer # imports the TF-IDF tool from scikit-learn.

def create_tfidf_features(text_data, max_features=5000):
    """
    Converts cleaned text data into TF-IDF numerical features.
    
    Args:
        text_data: A panda Series or list containing cleaned text.
        max_features: Max no. of TF-IDF features to keep.
        
    Returns:
        X: Tf_IDF features matrix.
        vectorizer: Fitted TF_IDF vectorizer.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english", #Remove common words like the, is, are, and, of.
        ngram_range=(1,2), #Use single words and two-word phrases.
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )

    X = vectorizer.fit_transform(text_data) # fit-> learn vocab. from cleaned text, transform -> converts each filing into numerical fearues
    # X - numerical matrix
    return X, vectorizer