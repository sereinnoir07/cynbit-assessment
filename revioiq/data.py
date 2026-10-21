"""Dataset loading, preprocessing, and synthetic data ingestion for RevioIQ.

Supports Yelp Full Review dataset with Amazon Polarity fallback.
Maps star ratings to 3 classes: negative, neutral, positive.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from datasets import load_dataset

from revioiq.config import DATA_DIR, SYNTHETIC_DATA_PATH

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def map_yelp_label(label: int) -> str:
    """Map Yelp 0-4 labels (1-5 stars) to 3-class sentiment."""
    if label in (0, 1):
        return "negative"
    elif label == 2:
        return "neutral"
    elif label in (3, 4):
        return "positive"
    return "neutral"


def map_amazon_label(label: int) -> str:
    """Map Amazon Polarity 0-1 labels to sentiment."""
    return "negative" if label == 0 else "positive"


def load_raw_hf_dataset():
    """Load review dataset from Hugging Face, trying Yelp with Amazon fallback."""
    try:
        # Try canonical Yelp datasets
        for name in ("Yelp/yelp_review_full", "yelp_review_full"):
            try:
                ds = load_dataset(name)
                return ds, "yelp"
            except Exception:
                continue
        raise RuntimeError("Yelp dataset not reachable")
    except Exception:
        # Fallback to amazon polarity
        for name in ("amazon_polarity", "fancyzhx/amazon_polarity"):
            try:
                ds = load_dataset(name)
                return ds, "amazon"
            except Exception:
                continue
        raise RuntimeError("Failed to load both Yelp and Amazon Polarity datasets from HF")


def get_train_test_datasets(
    n_train: int = 20000,
    n_test: int = 2000,
    seed: int = 42,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve seeded subsets of train (20k) and test (2k) reviews.

    Caches local CSVs in data/cache/ (gitignored) for fast repeatable execution.
    """
    train_cache = CACHE_DIR / f"train_subset_{n_train}_seed_{seed}.csv"
    test_cache = CACHE_DIR / f"test_subset_{n_test}_seed_{seed}.csv"

    if not force_refresh and train_cache.exists() and test_cache.exists():
        df_train = pd.read_csv(train_cache)
        df_test = pd.read_csv(test_cache)
        return df_train, df_test

    ds, ds_type = load_raw_hf_dataset()

    raw_train = ds["train"].shuffle(seed=seed).select(range(min(n_train, len(ds["train"]))))
    raw_test = ds["test"].shuffle(seed=seed).select(range(min(n_test, len(ds["test"]))))

    train_texts = raw_train["text"]
    test_texts = raw_test["text"]

    if ds_type == "yelp":
        train_labels = [map_yelp_label(lbl) for lbl in raw_train["label"]]
        test_labels = [map_yelp_label(lbl) for lbl in raw_test["label"]]
    else:
        train_labels = [map_amazon_label(lbl) for lbl in raw_train["label"]]
        test_labels = [map_amazon_label(lbl) for lbl in raw_test["label"]]

    df_train = pd.DataFrame({"text": train_texts, "sentiment": train_labels})
    df_test = pd.DataFrame({"text": test_texts, "sentiment": test_labels})

    df_train.to_csv(train_cache, index=False)
    df_test.to_csv(test_cache, index=False)

    return df_train, df_test


def load_synthetic_reviews() -> pd.DataFrame:
    """Load the 40-review multi-branch synthetic benchmark dataset."""
    if not SYNTHETIC_DATA_PATH.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at {SYNTHETIC_DATA_PATH}")
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    # Ensure boolean type for crisis_flag
    df["crisis_flag"] = df["crisis_flag"].astype(bool)
    return df
