"""TF-IDF + Logistic Regression sentiment classification baseline for RevioIQ.

Classifies customer reviews into: negative, neutral, positive.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from revioiq.config import SENTIMENT_MODEL_PATH


class SentimentClassifier:
    """Wrapper for TF-IDF + Logistic Regression sentiment pipeline."""

    def __init__(self, pipeline: Optional[Pipeline] = None):
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=30000,
                            ngram_range=(1, 2),
                            sublinear_tf=True,
                            min_df=2,
                            strip_accents="unicode",
                        ),
                    ),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            C=1.0,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            )
        self.classes_ = ["negative", "neutral", "positive"]

    def fit(self, texts: List[str], labels: List[str]) -> "SentimentClassifier":
        """Fit the TF-IDF vectorizer and Logistic Regression classifier."""
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        return self

    def predict(self, texts: Union[str, List[str]]) -> List[str]:
        """Predict sentiment labels for single or multiple texts."""
        if isinstance(texts, str):
            texts = [texts]
        preds = self.pipeline.predict(texts)
        return list(preds)

    def predict_proba(self, texts: Union[str, List[str]]) -> List[Dict[str, float]]:
        """Predict sentiment class probabilities."""
        if isinstance(texts, str):
            texts = [texts]
        probs = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.classes_)
        results = []
        for row in probs:
            results.append({classes[i]: float(row[i]) for i in range(len(classes))})
        return results

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save pipeline artifact to disk."""
        save_path = Path(path) if path else SENTIMENT_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "SentimentClassifier":
        """Load pipeline artifact from disk."""
        load_path = Path(path) if path else SENTIMENT_MODEL_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"Trained model not found at {load_path}")
        pipeline = joblib.load(load_path)
        inst = cls(pipeline=pipeline)
        inst.classes_ = list(pipeline.classes_)
        return inst


_GLOBAL_CLASSIFIER: Optional[SentimentClassifier] = None


def get_sentiment_classifier() -> SentimentClassifier:
    """Singleton getter for SentimentClassifier."""
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        if SENTIMENT_MODEL_PATH.exists():
            _GLOBAL_CLASSIFIER = SentimentClassifier.load(SENTIMENT_MODEL_PATH)
        else:
            _GLOBAL_CLASSIFIER = SentimentClassifier()
    return _GLOBAL_CLASSIFIER
