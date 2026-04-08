"""
Stage 3: Price consolidation.

The previous two stages decided WHICH listings refer to the same product.
This stage decides WHAT to display for each unified product:

  - the canonical name (we pick the most "complete" name in the cluster)
  - the lowest price found across all listings in the cluster
  - the highest price (for context - "save X% vs the most expensive seller")
  - the number of merged listings
  - the original listing IDs (so a real frontend could link to each seller)

Stage 3 is intentionally simple: by the time we get here, the hard work of
deciding what's a duplicate is already done. All that's left is the
straightforward "group-by + min(price)" the assignment literally asks for.
"""

import pandas as pd


def pick_canonical_name(names: list[str]) -> str:
    """
    Pick the best display name for a cluster of duplicate listings.

    Heuristic: prefer the longest name. Longer names tend to include more
    specifying detail (storage, color, generation), which is exactly what
    a price-comparison page wants to show.

    Tie-break: the alphabetically first name, just to keep the choice
    deterministic across runs.
    """
    return max(names, key=lambda n: (len(n), -ord(n[0]) if n else 0))


def consolidate(df: pd.DataFrame, final_clusters: list[set[int]]) -> pd.DataFrame:
    """
    Turn the clustered output of Stage 2 into a unified product catalog.

    Args:
        df: the original products DataFrame (must have product_id,
            product_name, price, category columns).
        final_clusters: list of sets of df row indices, one set per
            unified product. Singletons (1-element sets) are kept as-is -
            a unique product is still a product.

    Returns:
        DataFrame with one row per unified product, sorted by category
        then by canonical name.
    """
    rows = []
    for cluster in final_clusters:
        indices = sorted(cluster)
        sub = df.iloc[indices]

        names = sub["product_name"].tolist()
        prices = sub["price"].tolist()
        canonical_name = pick_canonical_name(names)

        min_price = int(min(prices))
        max_price = int(max(prices))
        savings_pct = round((1 - min_price / max_price) * 100, 1) if max_price > 0 else 0.0

        rows.append({
            "canonical_name": canonical_name,
            "category": sub.iloc[0]["category"],
            "min_price": min_price,
            "max_price": max_price,
            "savings_pct": savings_pct,
            "num_listings": len(indices),
            "listing_ids": ",".join(str(int(i)) for i in sub["product_id"].tolist()),
            "all_names": " | ".join(names),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["category", "canonical_name"]).reset_index(drop=True)
    return out


def print_summary(unified: pd.DataFrame) -> None:
    """Print a short human-readable summary of the unified catalog."""
    n_total = len(unified)
    n_multi = int((unified["num_listings"] > 1).sum())
    n_single = n_total - n_multi
    total_listings = int(unified["num_listings"].sum())

    print("=" * 60)
    print("STAGE 3: PRICE CONSOLIDATION")
    print("=" * 60)
    print(f"Input listings:    {total_listings}")
    print(f"Unified products:  {n_total}")
    print(f"  multi-listing:   {n_multi}")
    print(f"  single-listing:  {n_single}")

    multi = unified[unified["num_listings"] > 1].sort_values("savings_pct", ascending=False)
    if not multi.empty:
        print(f"\nTop 5 biggest price spreads:")
        for _, row in multi.head(5).iterrows():
            print(f"  {row['canonical_name'][:50]:50}  "
                  f"₪{row['min_price']:>5} – ₪{row['max_price']:>5}  "
                  f"(save {row['savings_pct']:.0f}%, {row['num_listings']} listings)")


if __name__ == "__main__":
    # Smoke test: run the full pipeline end-to-end on the synthetic dataset.
    from embeddings import run_embedding_stage
    from llm_refine_v2 import refine_clusters_v2

    df, clusters, singletons, _, embeddings = run_embedding_stage()
    final, _ = refine_clusters_v2(df, embeddings, clusters, singletons)
    unified = consolidate(df, final)
    print_summary(unified)

    from config import OUTPUT_CSV
    unified.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_CSV}")
