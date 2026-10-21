"""Configuration and environment management for RevioIQ.

Loads environment variables securely via python-dotenv.
Never exposes or logs secret values.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# Model configuration
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_GEMINI_MODEL = "gemini-2.5-flash-lite"

# Synthetic dataset path
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_reviews.csv"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"


def get_gemini_api_key() -> str:
    """Return GEMINI_API_KEY from environment or empty string.

    Never logs or prints the key.
    """
    return os.getenv("GEMINI_API_KEY", "").strip()


def has_gemini_api_key() -> bool:
    """Check if a non-empty Gemini API key is configured."""
    key = get_gemini_api_key()
    return bool(key) and key != "your_gemini_api_key_here"
