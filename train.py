"""Training script for RevioIQ TF-IDF + Logistic Regression sentiment baseline.

Trains on seeded 20,000 Yelp review subset and evaluates on 2,000 test reviews.
Saves model artifact to models/sentiment_pipeline.joblib.
"""

import sys
import time
from sklearn.metrics import accuracy_score, classification_report, f1_score

from revioiq.config import SENTIMENT_MODEL_PATH
from revioiq.data import get_train_test_datasets
from revioiq.sentiment import SentimentClassifier


def main():
    print("=" * 60)
    print("RevioIQ: Training Sentiment Analysis Baseline")
    print("=" * 60)

    start_time = time.time()
    print("Loading Yelp datasets (20,000 train / 2,000 test)...")
    df_train, df_test = get_train_test_datasets(n_train=20000, n_test=2000, seed=42)
    print(f"Loaded {len(df_train)} train samples and {len(df_test)} test samples.")
    print("Train label distribution:")
    print(df_train["sentiment"].value_counts().to_string())

    print("\nFitting TF-IDF + Logistic Regression pipeline...")
    clf = SentimentClassifier()
    fit_start = time.time()
    clf.fit(df_train["text"].tolist(), df_train["sentiment"].tolist())
    fit_time = time.time() - fit_start
    print(f"Pipeline fitted in {fit_time:.2f}s.")

    saved_path = clf.save(SENTIMENT_MODEL_PATH)
    print(f"Model saved successfully to: {saved_path}")

    print("\nEvaluating on 2,000 test reviews...")
    preds = clf.predict(df_test["text"].tolist())
    y_test = df_test["sentiment"].tolist()

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"\nHeadline Test Metrics:")
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds, digits=4))

    total_time = time.time() - start_time
    print(f"Total training workflow completed in {total_time:.2f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
