"""
Configuration for the product deduplication pipeline.
All tunable parameters live here - nothing is hardcoded in the logic.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root regardless of where Python was launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# --- Embedding Stage ---
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Similarity threshold for candidate pairs (Pass 1: pairwise comparison).
# 0.87 catches most duplicates while keeping clusters manageable.
SIMILARITY_THRESHOLD = 0.87

# Threshold for Pass 2: comparing singletons against cluster centroids.
# Lower than Pass 1 because centroids are more stable representations -
# the average of a cluster smooths out individual naming quirks.
CENTROID_THRESHOLD = 0.65

# --- LLM Stage ---
# Accept either OPENAI_API_KEY (standard) or API_KEY (project convention).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
LLM_MODEL = "gpt-4o-mini"

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_CSV = os.path.join(DATA_DIR, "products.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "deduplicated_products.csv")
