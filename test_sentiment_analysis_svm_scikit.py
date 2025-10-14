"""
Unit test for the SVM Sentiment Analysis example.
Ensures the model can train and predict without errors.
"""

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def tests_model_prediction():
    """Train an SVM and check predictions."""
    df = pd.DataFrame({
        "text": ["I love this!", "This is bad.", "It's okay."],
        "label": [2, 0, 1]  # 2=Positive, 0=Negative, 1=Neutral
    })

    model = make_pipeline(TfidfVectorizer(), LinearSVC())
    model.fit(df["text"], df["label"])

    preds = model.predict(["Awesome product!", "Horrible experience"])
    assert all(p in [0, 1, 2] for p in preds)

    print("Unit test passed successfully.")
