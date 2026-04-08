"""
Generate visualizations for the embedding tuning documentation.

Produces three charts:
1. `similarity_distribution.png` - density plot comparing the similarity-score
   distribution for real duplicate pairs vs non-duplicate pairs. Normalized
   so the two populations are directly comparable regardless of size.
2. `threshold_tradeoff.png` - grouped bars: for each threshold tried, how
   many real duplicates get caught vs how many false merges get introduced.
3. `cluster_network.png` - t-SNE projection of the product embeddings with
   each product colored by its embedding-stage cluster. Visualizes exactly
   why some clusters need LLM splitting: iPhones and Samsungs form big
   blobs that mix multiple variants.
"""

import sys
import os

# Make sure prints with Hebrew paths don't crash on Windows cp1252 stdout.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL, INPUT_CSV
from generate_dataset import PRODUCTS

matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["font.size"] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "approach", "images")


def build_ground_truth_map():
    """
    Build a map: product_name -> canonical_id.
    Two product names are 'true duplicates' iff their canonical_id matches.
    Pulled from generate_dataset.py PRODUCTS list - the synthetic ground truth.
    """
    name_to_canonical = {}
    for canonical_id, product in enumerate(PRODUCTS):
        for variation in product["variations"]:
            name_to_canonical[variation] = canonical_id
    return name_to_canonical


def compute_labeled_pair_scores(df, embeddings, name_to_canonical):
    """
    For every within-category product pair, compute (similarity, is_duplicate)
    where is_duplicate is True iff both names map to the same canonical product.
    """
    duplicate_scores = []
    non_duplicate_scores = []

    for category in df["category"].unique():
        indices = df[df["category"] == category].index.tolist()
        group_emb = embeddings[indices]
        sim_matrix = cosine_similarity(group_emb)
        for i in range(len(indices)):
            name_i = df.iloc[indices[i]]["product_name"]
            canon_i = name_to_canonical.get(name_i)
            for j in range(i + 1, len(indices)):
                name_j = df.iloc[indices[j]]["product_name"]
                canon_j = name_to_canonical.get(name_j)
                score = float(sim_matrix[i, j])
                if canon_i is not None and canon_j is not None and canon_i == canon_j:
                    duplicate_scores.append(score)
                else:
                    non_duplicate_scores.append(score)

    return np.array(duplicate_scores), np.array(non_duplicate_scores)


def build_clusters_at_threshold(df, embeddings, threshold):
    """Build clusters at a given threshold, returns (clusters, singletons)."""
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for category in df["category"].unique():
        indices = df[df["category"] == category].index.tolist()
        group_emb = embeddings[indices]
        sim_matrix = cosine_similarity(group_emb)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if sim_matrix[i, j] >= threshold:
                    union(indices[i], indices[j])

    clusters_dict = {}
    for i in range(len(df)):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = set()
        clusters_dict[root].add(i)

    clusters = [c for c in clusters_dict.values() if len(c) > 1]
    singletons = [c for c in clusters_dict.values() if len(c) == 1]
    return clusters, singletons


def centroid_pass(df, embeddings, clusters, singletons, threshold=0.65):
    """Pass 2: assign singletons to clusters via centroid matching."""
    if not singletons or not clusters:
        return clusters, singletons

    cluster_info = []
    for cluster in clusters:
        indices = list(cluster)
        centroid = embeddings[indices].mean(axis=0)
        categories = set(df.iloc[idx]["category"] for idx in indices)
        cluster_info.append({"centroid": centroid, "categories": categories})

    remaining = []
    assigned = 0

    for singleton_set in singletons:
        idx = list(singleton_set)[0]
        singleton_emb = embeddings[idx]
        singleton_cat = df.iloc[idx]["category"]

        best_score = -1
        best_ci = -1

        for ci, info in enumerate(cluster_info):
            if singleton_cat not in info["categories"]:
                continue
            score = float(cosine_similarity(
                singleton_emb.reshape(1, -1),
                info["centroid"].reshape(1, -1),
            )[0, 0])
            if score > best_score:
                best_score = score
                best_ci = ci

        if best_score >= threshold and best_ci >= 0:
            clusters[best_ci].add(idx)
            new_indices = list(clusters[best_ci])
            cluster_info[best_ci]["centroid"] = embeddings[new_indices].mean(axis=0)
            assigned += 1
        else:
            remaining.append(singleton_set)

    return clusters, remaining, assigned


def chart_1_labeled_distribution(duplicate_scores, non_duplicate_scores):
    """
    Density plot of similarity scores, split by ground truth.

    Previous version plotted raw counts, which was unreadable: the 1,490
    non-duplicate pairs completely buried the 201 duplicate pairs. This
    version normalizes each group to its own density so they're directly
    comparable, and drops the second threshold line / gap annotation so
    the eye has one thing to focus on: how much the two populations overlap
    around 0.87.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))

    bins = np.linspace(0, 1, 36)

    # `density=True` normalizes each histogram to integrate to 1, so the
    # visual comparison isn't dominated by the fact that there are 7x more
    # non-duplicate pairs than duplicate pairs.
    ax.hist(non_duplicate_scores, bins=bins, density=True,
            color="#C0392B", alpha=0.55, edgecolor="white", linewidth=0.5,
            label=f"Non-duplicates  ({len(non_duplicate_scores):,} pairs)")
    ax.hist(duplicate_scores, bins=bins, density=True,
            color="#27AE60", alpha=0.70, edgecolor="white", linewidth=0.5,
            label=f"Real duplicates  ({len(duplicate_scores)} pairs)")

    # One threshold line, clearly labeled.
    ax.axvline(x=0.87, color="#1B5E20", linewidth=2.2, linestyle="--")
    ymax = ax.get_ylim()[1]
    ax.text(0.87, ymax * 1.01, "threshold = 0.87",
            color="#1B5E20", fontsize=11, fontweight="bold",
            ha="center", va="bottom")

    ax.set_xlabel("Cosine similarity of a pair of product names", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Real duplicates score higher, but the two populations overlap.\n"
                 "Threshold 0.87 sacrifices some recall for a big precision gain.",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlim(0, 1.02)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "similarity_distribution.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def compute_tradeoff_at_thresholds(duplicate_scores, non_duplicate_scores, thresholds):
    """
    For each threshold, return (caught, missed, false_merges).
    - caught       = duplicate pairs whose score >= threshold (true positives)
    - missed       = duplicate pairs whose score < threshold (false negatives)
    - false_merges = non-duplicate pairs whose score >= threshold (false positives)
    """
    rows = []
    total_dupes = len(duplicate_scores)
    for t in thresholds:
        caught = int((duplicate_scores >= t).sum())
        missed = total_dupes - caught
        false_merges = int((non_duplicate_scores >= t).sum())
        rows.append({
            "threshold": t,
            "caught": caught,
            "missed": missed,
            "false_merges": false_merges,
            "recall_pct": (caught / total_dupes * 100) if total_dupes else 0,
        })
    return rows


def chart_2_threshold_tradeoff(duplicate_scores, non_duplicate_scores):
    """
    Grouped bar chart: for each threshold, show
      - duplicates caught (green, bigger = better)
      - false merges introduced (red, smaller = better)
    """
    thresholds = [0.65, 0.75, 0.82, 0.87, 0.92]
    rows = compute_tradeoff_at_thresholds(duplicate_scores, non_duplicate_scores, thresholds)

    x = np.arange(len(thresholds))
    width = 0.38

    caught = [r["caught"] for r in rows]
    false_merges = [r["false_merges"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_caught = ax.bar(x - width / 2, caught, width,
                         color="#27AE60", alpha=0.9, edgecolor="white",
                         label="Real duplicates caught (higher = better)")
    bars_false = ax.bar(x + width / 2, false_merges, width,
                        color="#C0392B", alpha=0.9, edgecolor="white",
                        label="False merges introduced (lower = better)")

    # Annotate bar values
    for bars in (bars_caught, bars_false):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + max(caught + false_merges) * 0.012,
                    f"{int(h)}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in thresholds], fontsize=11)
    ax.set_xlabel("Similarity threshold", fontsize=11)
    ax.set_ylabel("Number of product pairs", fontsize=11)
    ax.set_title("The threshold tradeoff: catching duplicates vs introducing false merges\n"
                 "(at 0.65 the false merges DROWN the real ones - at 0.87 the ratio flips)",
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Highlight the chosen threshold
    chosen_idx = thresholds.index(0.87)
    ax.axvspan(chosen_idx - 0.5, chosen_idx + 0.5, color="#27AE60", alpha=0.08)
    ax.text(chosen_idx, ax.get_ylim()[1] * 0.93, "CHOSEN",
            ha="center", fontsize=10, fontweight="bold", color="#1B5E20",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D5F5E3",
                      edgecolor="#1B5E20", linewidth=1.2))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "threshold_tradeoff.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")

    # Also print the table to stdout so it can be pasted into the docs
    print("\nThreshold tradeoff table:")
    print(f"{'threshold':>10} | {'caught':>7} | {'missed':>7} | {'false merges':>13} | {'recall':>7}")
    print("-" * 60)
    for r in rows:
        print(f"{r['threshold']:>10.2f} | {r['caught']:>7d} | {r['missed']:>7d} | "
              f"{r['false_merges']:>13d} | {r['recall_pct']:>6.1f}%")
    return rows


def chart_3_cluster_network(df, embeddings, name_to_canonical):
    """
    Visualize the embedding stage's output in 2D.

    Method:
      1. Run the actual two-pass embedding clustering at threshold 0.87
         (plus centroid rescue at 0.65) to get the same clusters the
         pipeline sees.
      2. Project the 768-dim embeddings to 2D with t-SNE for plotting.
      3. Color each product by its cluster id; singletons are light gray.
      4. Draw thin edges between within-category pairs whose similarity is
         >= 0.87 - the edges that actually built the clusters.
      5. Annotate a handful of the problematic multi-variant clusters so
         the reader can see the exact failure modes the LLM has to fix.

    This chart answers one question visually: *what does the input to
    the LLM stage actually look like?*
    """
    from sklearn.manifold import TSNE

    # ---- Build the real embedding-stage clusters ----
    clusters, singletons = build_clusters_at_threshold(df, embeddings, 0.87)
    clusters, singletons, _ = centroid_pass(df, embeddings, clusters, singletons, threshold=0.65)

    n = len(df)
    cluster_of = [-1] * n
    for ci, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_of[idx] = ci
    # singletons keep cluster_of[idx] = -1

    # ---- 2D projection ----
    print("  Running t-SNE on embeddings (this takes a few seconds)...")
    tsne = TSNE(n_components=2, perplexity=15, random_state=42,
                init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(13, 9))

    # Edges first so dots draw on top
    edge_count = 0
    for category in df["category"].unique():
        indices = df[df["category"] == category].index.tolist()
        group_emb = embeddings[indices]
        sim = cosine_similarity(group_emb)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if sim[i, j] >= 0.87:
                    a, b = indices[i], indices[j]
                    ax.plot([coords[a, 0], coords[b, 0]],
                            [coords[a, 1], coords[b, 1]],
                            color="#BBBBBB", linewidth=0.6, alpha=0.55,
                            zorder=1)
                    edge_count += 1
    print(f"  Drew {edge_count} similarity edges (within-category, sim >= 0.87)")

    # Curated vivid palette: no grays (singletons own gray) and no yellows
    # (the annotation boxes own yellow). 24 visually distinct hues so each
    # cluster reads as its own thing.
    CLUSTER_COLORS = [
        "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
        "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
        "#DCBEFF", "#9A6324", "#800000", "#AAFFC3", "#808000",
        "#FFD8B1", "#000075", "#FF1493", "#20B2AA", "#B22222",
        "#2E8B57", "#4B0082", "#D2691E", "#1ABC9C",
    ]
    n_clusters = len(clusters)
    cluster_color = lambda ci: CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]

    # Singletons
    for i in range(n):
        if cluster_of[i] == -1:
            ax.scatter(coords[i, 0], coords[i, 1], s=70,
                       color="#CCCCCC", edgecolors="#666666", linewidths=0.6,
                       zorder=2)
    # Clustered points
    for i in range(n):
        ci = cluster_of[i]
        if ci == -1:
            continue
        ax.scatter(coords[i, 0], coords[i, 1], s=95,
                   color=cluster_color(ci), edgecolors="white",
                   linewidths=0.8, zorder=3)

    # ---- Annotate the problematic multi-variant clusters ----
    # For each cluster, count how many distinct canonical products it contains.
    # Clusters that mix >=2 canonical products are the ones the LLM has to
    # split - those are what we want to label.
    def canonical_ids_in_cluster(cluster):
        ids = set()
        for idx in cluster:
            canon = name_to_canonical.get(df.iloc[idx]["product_name"])
            if canon is not None:
                ids.add(canon)
        return ids

    annotated = 0
    problematic = []
    for ci, cluster in enumerate(clusters):
        canons = canonical_ids_in_cluster(cluster)
        if len(canons) >= 2:
            problematic.append((ci, cluster, len(canons)))
    # Sort by size so we label the biggest mega-clusters first
    problematic.sort(key=lambda t: -len(t[1]))

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    label_padding = (ymax - ymin) * 0.04

    for ci, cluster, n_canon in problematic[:6]:
        idxs = list(cluster)
        cx = float(np.mean([coords[i, 0] for i in idxs]))
        cy = float(np.mean([coords[i, 1] for i in idxs]))
        # Use the first listing's category + count + variant summary
        size = len(cluster)
        # Find a short descriptive label from the cluster contents
        names = [df.iloc[i]["product_name"] for i in idxs]
        label = _short_cluster_label(names, size, n_canon)
        ax.annotate(label,
                    xy=(cx, cy),
                    xytext=(cx, cy + label_padding),
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#2C3E50",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#FFFACD",
                              edgecolor="#B8860B",
                              linewidth=1.0, alpha=0.95),
                    zorder=4)
        annotated += 1

    ax.set_title(f"Embedding-stage clusters in 2D (t-SNE of {n} product names)\n"
                 f"Dots = products, colors = cluster, lines = cosine >= 0.87. "
                 f"Yellow labels mark clusters that mix >=2 variants and need LLM splitting.",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend: explain the color scheme
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CLUSTER_COLORS[0], edgecolor="white",
              label=f"Clustered ({n - sum(1 for c in cluster_of if c == -1)} products "
                    f"in {n_clusters} clusters)"),
        Patch(facecolor="#CCCCCC", edgecolor="#666666",
              label=f"Singleton ({sum(1 for c in cluster_of if c == -1)} products)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=10, framealpha=0.95)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cluster_network.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def _short_cluster_label(names: list[str], size: int, n_canon: int) -> str:
    """
    Produce a short human-readable label for an annotated cluster.

    We look for recognizable anchor words (iPhone, Galaxy, MacBook, ...)
    in the cluster's names and fall back to the first name's first two
    tokens otherwise. The count of items + the number of true canonical
    products is appended so the reader knows how bad the merge is.
    """
    joined = " ".join(names).lower()
    anchors = [
        ("iphone",         "iPhone family"),
        ("s24 ultra",      "Galaxy S24 Ultra"),
        ("galaxy s24",     "Galaxy S24 family"),
        ("galaxy s25",     "Galaxy S25 family"),
        ("macbook",        "MacBook Air family"),
        ("oled c4",        "LG OLED C4 family"),
        ("crystal uhd",    "Samsung Crystal UHD"),
        ("washing",        "Washing machines"),
        ("xm5",            "Sony XM family"),
        ("airpods",        "AirPods family"),
        ("vivobook",       "ASUS VivoBook"),
        ("xps",            "Dell XPS"),
        ("ideapad",        "Lenovo IdeaPad"),
    ]
    for needle, pretty in anchors:
        if needle in joined:
            return f"{pretty}\n{size} items → {n_canon} real products"
    first = names[0].split()[:2]
    return f"{' '.join(first)}\n{size} items → {n_canon} real products"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data and generate embeddings
    print("Loading data and generating embeddings...")
    df = pd.read_csv(INPUT_CSV)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(df["product_name"].tolist(), show_progress_bar=True, batch_size=32)

    # Compute labeled pair scores using ground truth from generate_dataset.py
    print("\nLabeling pairs against ground truth...")
    name_to_canonical = build_ground_truth_map()
    duplicate_scores, non_duplicate_scores = compute_labeled_pair_scores(
        df, embeddings, name_to_canonical
    )
    print(f"  Real duplicate pairs: {len(duplicate_scores)}")
    print(f"  Non-duplicate pairs:  {len(non_duplicate_scores):,}")

    # Chart 1: Labeled distribution (density plot)
    print("\nGenerating Chart 1: Similarity distribution (by ground truth)...")
    chart_1_labeled_distribution(duplicate_scores, non_duplicate_scores)

    # Chart 2: Threshold tradeoff bar chart
    print("Generating Chart 2: Threshold tradeoff...")
    chart_2_threshold_tradeoff(duplicate_scores, non_duplicate_scores)

    # Chart 3: Cluster network (t-SNE)
    print("Generating Chart 3: Cluster network (t-SNE)...")
    chart_3_cluster_network(df, embeddings, name_to_canonical)

    print(f"\nAll charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
