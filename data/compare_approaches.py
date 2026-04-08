"""
Apples-to-apples comparison: v1 (always-LLM) vs v2 (confidence-gated).

Runs the embedding stage once, then runs both refinement strategies on the
exact same input. Computes accuracy metrics against the synthetic ground
truth from generate_dataset.py and produces a comparison chart.

Metrics computed (all pair-level, restricted to same-category pairs):
  - Precision: of the pairs we put in the same cluster, how many are
    actually duplicates of each other?
  - Recall:    of the pairs that ARE duplicates of each other, how many
    did we manage to put in the same cluster?
  - F1:        harmonic mean of precision and recall.
  - Cluster purity: fraction of multi-item clusters that contain only
    items from a single canonical product.
  - Coverage:  fraction of canonical products that have at least one of
    their listings in a multi-item cluster.

API call counts come from the shared counter in llm_refine.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from embeddings import run_embedding_stage
from llm_refine import refine_clusters as refine_v1
from llm_refine import reset_api_call_count, get_api_call_count
from llm_refine_v2 import refine_clusters_v2
from generate_dataset import PRODUCTS

matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["font.size"] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "approach", "images")


# ---------------------------------------------------------------------------
# Ground truth + metrics
# ---------------------------------------------------------------------------

def build_ground_truth(df) -> list[int]:
    """
    Returns a list parallel to df where entry i is the canonical product id
    of df.iloc[i]. -1 if the name isn't in PRODUCTS (shouldn't happen for
    the synthetic dataset).
    """
    name_to_canonical = {}
    for cid, product in enumerate(PRODUCTS):
        for variation in product["variations"]:
            name_to_canonical[variation] = cid
    return [name_to_canonical.get(name, -1) for name in df["product_name"]]


def compute_metrics(df, predicted_clusters, ground_truth: list[int]) -> dict:
    n = len(df)
    cats = df["category"].tolist()

    # Build per-product cluster id (-1 if singleton/unassigned)
    pred_id = [-1] * n
    for ci, cluster in enumerate(predicted_clusters):
        for idx in cluster:
            pred_id[idx] = ci

    # Pair-level confusion matrix, restricted to within-category pairs
    tp = fp = fn = tn = 0
    for i in range(n):
        for j in range(i + 1, n):
            if cats[i] != cats[j]:
                continue
            same_pred = pred_id[i] != -1 and pred_id[i] == pred_id[j]
            same_true = ground_truth[i] != -1 and ground_truth[i] == ground_truth[j]
            if same_pred and same_true:
                tp += 1
            elif same_pred and not same_true:
                fp += 1
            elif not same_pred and same_true:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Cluster purity: % of multi-item clusters containing only one canonical id
    multi = [c for c in predicted_clusters if len(c) > 1]
    pure = 0
    for c in multi:
        canons = {ground_truth[i] for i in c if ground_truth[i] != -1}
        if len(canons) <= 1:
            pure += 1
    purity = pure / len(multi) if multi else 0.0

    # Coverage: % of canonical products that have ≥2 listings present in
    # the input AND are represented by a single multi-cluster.
    canonical_to_indices: dict[int, list[int]] = {}
    for i, c in enumerate(ground_truth):
        if c != -1:
            canonical_to_indices.setdefault(c, []).append(i)

    canonicals_with_dupes = [c for c, idxs in canonical_to_indices.items() if len(idxs) > 1]
    correctly_covered = 0
    for c in canonicals_with_dupes:
        idxs = canonical_to_indices[c]
        cluster_ids = {pred_id[i] for i in idxs}
        cluster_ids.discard(-1)
        if len(cluster_ids) == 1:
            correctly_covered += 1
    coverage = correctly_covered / len(canonicals_with_dupes) if canonicals_with_dupes else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "purity": purity,
        "coverage": coverage,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_clusters": len(multi),
        "n_singletons": sum(1 for c in predicted_clusters if len(c) == 1),
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def make_comparison_chart(v1: dict, v2: dict, v1_calls: int, v2_calls: int):
    fig, (ax_acc, ax_calls) = plt.subplots(1, 2, figsize=(13, 5.8),
                                           gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: grouped bar chart of accuracy metrics
    metric_names = ["Precision", "Recall", "F1", "Cluster purity", "Coverage"]
    metric_keys = ["precision", "recall", "f1", "purity", "coverage"]
    v1_vals = [v1[k] * 100 for k in metric_keys]
    v2_vals = [v2[k] * 100 for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.38

    bars1 = ax_acc.bar(x - width / 2, v1_vals, width,
                       color="#2E86AB", alpha=0.92, edgecolor="white",
                       label="v1 (always call LLM)")
    bars2 = ax_acc.bar(x + width / 2, v2_vals, width,
                       color="#27AE60", alpha=0.92, edgecolor="white",
                       label="v2 (confidence-gated)")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax_acc.text(b.get_x() + b.get_width() / 2, h + 1,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold")

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(metric_names, fontsize=10)
    ax_acc.set_ylabel("Score (%)", fontsize=11)
    ax_acc.set_ylim(0, 115)
    ax_acc.set_title("Accuracy: v1 vs v2 (higher is better)",
                     fontsize=12.5, fontweight="bold", pad=12)
    ax_acc.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax_acc.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax_acc.set_axisbelow(True)

    # Right: API call count comparison
    call_x = np.arange(2)
    call_vals = [v1_calls, v2_calls]
    colors = ["#2E86AB", "#27AE60"]
    bars_calls = ax_calls.bar(call_x, call_vals, width=0.55,
                              color=colors, alpha=0.92, edgecolor="white")
    for b, v in zip(bars_calls, call_vals):
        ax_calls.text(b.get_x() + b.get_width() / 2, v + 1.5,
                      f"{v}", ha="center", va="bottom",
                      fontsize=12, fontweight="bold")

    reduction = (1 - v2_calls / v1_calls) * 100 if v1_calls else 0
    ax_calls.set_xticks(call_x)
    ax_calls.set_xticklabels(["v1", "v2"], fontsize=11)
    ax_calls.set_ylabel("Number of LLM API calls", fontsize=11)
    ax_calls.set_ylim(0, max(call_vals) * 1.25)
    ax_calls.set_title(f"API calls (lower is better)\n"
                       f"v2 saves {reduction:.0f}%",
                       fontsize=12.5, fontweight="bold", pad=12)
    ax_calls.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax_calls.set_axisbelow(True)

    plt.suptitle("v1 vs v2: same accuracy, fewer API calls",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "v1_vs_v2_comparison.png")
    fig.savefig(path)
    plt.close()
    print(f"\nSaved chart: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Running embedding stage once (shared between v1 and v2)...")
    df, clusters, singletons, _, embeddings = run_embedding_stage()
    ground_truth = build_ground_truth(df)

    print("\n" + "#" * 70)
    print("# Running v1 (always-LLM)")
    print("#" * 70)
    reset_api_call_count()
    final_v1 = refine_v1(df, clusters, singletons)
    v1_calls = get_api_call_count()
    m_v1 = compute_metrics(df, final_v1, ground_truth)

    print("\n" + "#" * 70)
    print("# Running v2 (confidence-gated)")
    print("#" * 70)
    reset_api_call_count()
    final_v2, v2_stats = refine_clusters_v2(df, embeddings, clusters, singletons)
    v2_calls = get_api_call_count()
    m_v2 = compute_metrics(df, final_v2, ground_truth)

    # ---- Print comparison table ----
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    rows = [
        ("API calls", str(v1_calls), str(v2_calls)),
        ("Multi-item clusters", str(m_v1["n_clusters"]), str(m_v2["n_clusters"])),
        ("Unmatched singletons", str(m_v1["n_singletons"]), str(m_v2["n_singletons"])),
        ("Precision",  f"{m_v1['precision']*100:.2f}%", f"{m_v2['precision']*100:.2f}%"),
        ("Recall",     f"{m_v1['recall']*100:.2f}%",    f"{m_v2['recall']*100:.2f}%"),
        ("F1",         f"{m_v1['f1']*100:.2f}%",        f"{m_v2['f1']*100:.2f}%"),
        ("Cluster purity", f"{m_v1['purity']*100:.2f}%", f"{m_v2['purity']*100:.2f}%"),
        ("Coverage",       f"{m_v1['coverage']*100:.2f}%", f"{m_v2['coverage']*100:.2f}%"),
        ("True positive pairs", str(m_v1["tp"]), str(m_v2["tp"])),
        ("False positive pairs", str(m_v1["fp"]), str(m_v2["fp"])),
        ("False negative pairs", str(m_v1["fn"]), str(m_v2["fn"])),
    ]
    print(f"{'Metric':<25} {'v1 (always-LLM)':<22} {'v2 (gated)':<22}")
    print("-" * 70)
    for label, v1, v2 in rows:
        print(f"{label:<25} {v1:<22} {v2:<22}")

    print(f"\nv2 stats: {v2_stats}")
    print(f"\nv2 saved {(1 - v2_calls/v1_calls)*100:.1f}% of API calls "
          f"({v1_calls - v2_calls} fewer calls)")

    # ---- Make chart ----
    make_comparison_chart(m_v1, m_v2, v1_calls, v2_calls)


if __name__ == "__main__":
    main()
