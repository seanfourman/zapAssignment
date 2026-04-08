"""
Stage 1: Embedding-based candidate clustering.

Two-pass approach:
  Pass 1 - Pairwise comparison within categories at a high threshold (0.87).
           Catches obvious duplicates, builds initial clusters.
  Pass 2 - Compare leftover singletons against cluster centroids at a lower
           threshold (0.72). Catches harder cases like heavily transliterated
           Hebrew names that no single product in the cluster was close enough to.

Key insight: We filter by category FIRST, then cluster within each category.
This prevents transitive chaining across unrelated products.

Goal: HIGH RECALL - catch all possible duplicates, even if some false positives
slip through. The LLM stage will handle precision.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD, CENTROID_THRESHOLD, INPUT_CSV


def load_products(csv_path: str = INPUT_CSV) -> pd.DataFrame:
    """Load the product dataset from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} products from {csv_path}")
    return df


def generate_embeddings(product_names: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of product names.
    Uses a multilingual model that handles Hebrew + English natively.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Generating embeddings for {len(product_names)} products...")
    embeddings = model.encode(product_names, show_progress_bar=True, batch_size=32)

    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


def find_candidate_pairs_in_group(
    indices: list[int],
    embeddings: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """
    Find candidate pairs within a group of product indices.
    Only compares products within the same group (category).
    """
    pairs = []
    group_embeddings = embeddings[indices]
    sim_matrix = cosine_similarity(group_embeddings)

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if sim_matrix[i, j] >= threshold:
                pairs.append((indices[i], indices[j], float(sim_matrix[i, j])))

    return pairs


def build_clusters(n_products: int, pairs: list[tuple[int, int, float]]) -> list[set[int]]:
    """
    Build clusters from candidate pairs using Union-Find (connected components).
    Returns (multi-item clusters, singleton sets).
    """
    parent = list(range(n_products))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j, _ in pairs:
        union(i, j)

    clusters_dict = {}
    for i in range(n_products):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = set()
        clusters_dict[root].add(i)

    clusters = [c for c in clusters_dict.values() if len(c) > 1]
    singletons = [c for c in clusters_dict.values() if len(c) == 1]

    return clusters, singletons


def assign_singletons_to_clusters(
    singletons: list[set[int]],
    clusters: list[set[int]],
    embeddings: np.ndarray,
    df: pd.DataFrame,
    threshold: float = CENTROID_THRESHOLD,
) -> tuple[list[set[int]], list[set[int]]]:
    """
    Pass 2: Try to assign singletons to existing clusters by comparing
    each singleton's embedding against the centroid of each same-category cluster.

    Centroids (average embedding of all cluster members) are more stable than
    any individual product name - they smooth out naming quirks, making it safer
    to use a lower similarity threshold.
    """
    if not singletons or not clusters:
        return clusters, singletons

    # Precompute cluster centroids and their categories
    cluster_info = []
    for cluster in clusters:
        indices = list(cluster)
        centroid = embeddings[indices].mean(axis=0)
        categories = set(df.iloc[idx]["category"] for idx in indices)
        cluster_info.append({"centroid": centroid, "categories": categories})

    remaining_singletons = []
    assigned_count = 0

    for singleton_set in singletons:
        idx = list(singleton_set)[0]
        singleton_emb = embeddings[idx]
        singleton_cat = df.iloc[idx]["category"]

        best_score = -1
        best_cluster_idx = -1

        for ci, info in enumerate(cluster_info):
            # Only compare within same category
            if singleton_cat not in info["categories"]:
                continue

            score = float(cosine_similarity(
                singleton_emb.reshape(1, -1),
                info["centroid"].reshape(1, -1)
            )[0, 0])

            if score > best_score:
                best_score = score
                best_cluster_idx = ci

        if best_score >= threshold and best_cluster_idx >= 0:
            clusters[best_cluster_idx].add(idx)
            assigned_count += 1
            # Update centroid after adding new member
            new_indices = list(clusters[best_cluster_idx])
            cluster_info[best_cluster_idx]["centroid"] = embeddings[new_indices].mean(axis=0)
        else:
            remaining_singletons.append(singleton_set)

    print(f"  Pass 2: assigned {assigned_count} singletons to clusters, "
          f"{len(remaining_singletons)} remain unmatched")

    return clusters, remaining_singletons


def run_embedding_stage(csv_path: str = INPUT_CSV):
    """
    Run the full embedding stage:
    1. Load products
    2. Generate embeddings for all products at once (efficient)
    3. Pass 1: Find candidate pairs WITHIN each category at high threshold
    4. Build initial clusters
    5. Pass 2: Assign singletons to clusters via centroid matching at lower threshold
    """
    print("=" * 60)
    print("STAGE 1: EMBEDDING-BASED CANDIDATE CLUSTERING")
    print("=" * 60)

    # Load data
    df = load_products(csv_path)

    # Generate embeddings for ALL products in one batch
    product_names = df["product_name"].tolist()
    embeddings = generate_embeddings(product_names)

    # --- Pass 1: Pairwise comparison within categories ---
    print(f"\nPass 1: Pairwise comparison within categories (threshold={SIMILARITY_THRESHOLD})")
    all_pairs = []

    for category in df["category"].unique():
        cat_indices = df[df["category"] == category].index.tolist()
        pairs = find_candidate_pairs_in_group(cat_indices, embeddings)
        print(f"  {category}: {len(cat_indices)} products, {len(pairs)} candidate pairs")
        all_pairs.extend(pairs)

    print(f"  Total candidate pairs: {len(all_pairs)}")

    # Build initial clusters
    clusters, singletons = build_clusters(len(df), all_pairs)
    print(f"  Pass 1 result: {len(clusters)} clusters + {len(singletons)} singletons")

    # --- Pass 2: Assign singletons to clusters via centroids ---
    print(f"\nPass 2: Singleton-to-centroid matching (threshold={CENTROID_THRESHOLD})")
    clusters, remaining_singletons = assign_singletons_to_clusters(
        singletons, clusters, embeddings, df
    )

    # --- Print final results ---
    print(f"\n{'=' * 60}")
    print(f"FINAL: {len(clusters)} candidate clusters + {len(remaining_singletons)} unmatched singletons")
    print(f"{'=' * 60}")

    for i, cluster in enumerate(clusters):
        indices = sorted(cluster)
        print(f"\nCluster {i + 1} ({len(indices)} items):")
        for idx in indices:
            row = df.iloc[idx]
            print(f"  - [{row['product_id']}] {row['product_name']} (₪{row['price']})")

    if remaining_singletons:
        print(f"\n--- Unmatched singletons ---")
        for s in remaining_singletons:
            idx = list(s)[0]
            row = df.iloc[idx]
            print(f"  - [{row['product_id']}] {row['product_name']} ({row['category']})")

    return df, clusters, remaining_singletons, all_pairs, embeddings


if __name__ == "__main__":
    df, clusters, singletons, pairs, embeddings = run_embedding_stage()
