"""
Stage 2: LLM-based cluster refinement.

The embedding stage gives us candidate clusters with HIGH RECALL but imperfect
precision - different storage configurations or model variants get merged
together. This stage uses GPT-4o-mini to:

  1. Validate / split each candidate cluster
     "Are these really the same product? If not, group the ones that are."

  2. Match leftover singletons against existing clusters
     "Does this product belong in any of these groups?"

Goal: HIGH PRECISION - only merge what's truly the same product.

Why GPT-4o-mini: cheap, fast, good Hebrew understanding, and cluster validation
is a structured comparison task - not something that needs frontier reasoning.
"""

import json
from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL


_client = None

# Module-level API call counter so v1 and v2 can be compared apples-to-apples.
_api_call_count = 0


def reset_api_call_count() -> None:
    global _api_call_count
    _api_call_count = 0


def get_api_call_count() -> int:
    return _api_call_count


def get_client() -> OpenAI:
    """Lazy-init the OpenAI client so importing this module doesn't require a key."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "No OpenAI API key found. Set OPENAI_API_KEY or API_KEY in .env"
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Pass 1: split / validate candidate clusters
# ---------------------------------------------------------------------------

SPLIT_SYSTEM_PROMPT = """You are a product deduplication assistant for an Israeli price comparison site.

You will receive a candidate cluster of product listings that an embedding model
flagged as possible duplicates. Your job is to decide which listings refer to
EXACTLY the same product and which are different products that just have similar names.

Rules for "same product":
- Same brand, same model, same storage / size / color / generation.
- Hebrew and English names of the same product ARE the same product.
  Example: "Samsung Galaxy S24 Ultra 256GB" == "סמסונג גלקסי S24 אולטרה 256GB"
- Different storage (256GB vs 512GB) = DIFFERENT products.
- Different generation (XM4 vs XM5, iPhone 15 vs 16) = DIFFERENT products.
- Different model variant (S25 vs S25 FE, Pro vs Pro Max) = DIFFERENT products.
- Different size (55" vs 65") = DIFFERENT products.

Output format: a JSON object with a single key "groups", whose value is a list
of groups. Each group is a list of the listing IDs (integers) that belong together.
Every input ID must appear in exactly one group. Singletons are allowed (a group
with one ID means that listing is unique within this cluster).

Example output:
{"groups": [[1, 3, 5], [2, 4], [6]]}

Return ONLY valid JSON, nothing else."""


def split_cluster_with_llm(cluster_items: list[dict]) -> list[list[int]]:
    """
    Ask the LLM to split one candidate cluster into groups of true duplicates.

    cluster_items: list of {"id": int, "name": str} dicts.
    Returns: list of groups, each group a list of ids.
    """
    if len(cluster_items) <= 1:
        return [[item["id"] for item in cluster_items]]

    user_content = "Candidate cluster:\n" + "\n".join(
        f'  {item["id"]}: {item["name"]}' for item in cluster_items
    )

    client = get_client()
    global _api_call_count
    _api_call_count += 1
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SPLIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        groups = parsed.get("groups", [])
        # Sanity check: every input id must appear exactly once
        seen = {gid for g in groups for gid in g}
        expected = {item["id"] for item in cluster_items}
        if seen != expected:
            # LLM dropped or hallucinated ids - fall back to keeping cluster intact
            print(f"  Warning: LLM output mismatch, keeping cluster intact")
            return [list(expected)]
        return groups
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  Warning: failed to parse LLM response ({e}), keeping cluster intact")
        return [[item["id"] for item in cluster_items]]


# ---------------------------------------------------------------------------
# Pass 2: match singletons against existing clusters
# ---------------------------------------------------------------------------

MATCH_SYSTEM_PROMPT = """You are a product deduplication assistant for an Israeli price comparison site.

You will receive ONE unmatched product listing and a numbered list of existing
product groups (each group is a set of listings already known to be the same product).

Your job: decide whether the unmatched listing is the SAME product as any of the
existing groups. Use the same strict rules:
- Hebrew and English names of the same product ARE the same product.
- Different storage / generation / model variant / size = DIFFERENT products.

Output format: a JSON object with a single key "match_group", whose value is
either the group number (1-indexed integer) the listing belongs to, or null if
it doesn't match any group.

Example outputs:
{"match_group": 3}
{"match_group": null}

Return ONLY valid JSON, nothing else."""


def match_singleton_with_llm(
    singleton_name: str,
    candidate_groups: list[list[str]],
) -> int | None:
    """
    Ask the LLM whether a singleton belongs to any of the candidate groups.

    candidate_groups: list of groups, each group a list of product names.
    Returns: 0-indexed position into candidate_groups, or None.
    """
    if not candidate_groups:
        return None

    groups_text = "\n".join(
        f"Group {i + 1}:\n" + "\n".join(f"  - {name}" for name in group)
        for i, group in enumerate(candidate_groups)
    )

    user_content = (
        f"Unmatched listing: {singleton_name}\n\n"
        f"Existing groups:\n{groups_text}"
    )

    client = get_client()
    global _api_call_count
    _api_call_count += 1
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": MATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        match = parsed.get("match_group")
        if match is None:
            return None
        idx = int(match) - 1  # convert 1-indexed to 0-indexed
        if 0 <= idx < len(candidate_groups):
            return idx
        return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"  Warning: failed to parse LLM match response ({e})")
        return None


# ---------------------------------------------------------------------------
# Pass 3: merge clusters that refer to the same product but ended up split
# ---------------------------------------------------------------------------

def merge_split_clusters(df, multi_clusters: list[set[int]]) -> tuple[list[set[int]], int]:
    """
    Pass 3: catch the case where the SAME product ended up in two separate
    clusters (e.g. an English-named cluster and a Hebrew-named cluster).

    For each cluster, we treat its representative item as a "singleton" and
    ask the LLM whether it matches any OTHER cluster in the same category.
    Union-Find handles transitive merges in a single pass.
    """
    if len(multi_clusters) < 2:
        return multi_clusters, 0

    clusters = [set(c) for c in multi_clusters]
    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[max(px, py)] = min(px, py)

    merge_count = 0

    for i in range(len(clusters)):
        rep_idx = sorted(clusters[i])[0]
        rep_name = df.iloc[rep_idx]["product_name"]
        rep_cat = df.iloc[rep_idx]["category"]

        # Candidate other clusters: same category, not already merged with i.
        candidates: list[list[str]] = []
        candidate_indices: list[int] = []
        for j in range(len(clusters)):
            if j == i or find(i) == find(j):
                continue
            j_cats = {df.iloc[k]["category"] for k in clusters[j]}
            if rep_cat not in j_cats:
                continue
            candidates.append([df.iloc[k]["product_name"] for k in clusters[j]])
            candidate_indices.append(j)

        if not candidates:
            continue

        match_idx = match_singleton_with_llm(rep_name, candidates)
        if match_idx is not None:
            target_j = candidate_indices[match_idx]
            union(i, target_j)
            merge_count += 1
            print(f"  Merged cluster {i} ('{rep_name}') -> cluster {target_j}")

    # Coalesce clusters according to union-find roots.
    merged: dict[int, set[int]] = {}
    for i, cluster in enumerate(clusters):
        root = find(i)
        merged.setdefault(root, set()).update(cluster)

    return list(merged.values()), merge_count


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def refine_clusters(df, clusters, singletons):
    """
    Run the full LLM refinement stage.

    Args:
        df: products DataFrame (must have product_id, product_name columns).
        clusters: list of sets of df indices (from embedding stage).
        singletons: list of single-element sets of df indices.

    Returns:
        final_clusters: list of sets of df indices, after splitting and singleton matching.
    """
    print("=" * 60)
    print("STAGE 2: LLM REFINEMENT")
    print("=" * 60)
    print(f"Input: {len(clusters)} candidate clusters, {len(singletons)} singletons")

    # --- Pass 1: split each candidate cluster ---
    print(f"\nPass 1: validating/splitting {len(clusters)} clusters with {LLM_MODEL}")
    refined_clusters: list[set[int]] = []

    for ci, cluster in enumerate(clusters, 1):
        indices = sorted(cluster)
        items = [
            {"id": int(idx), "name": df.iloc[idx]["product_name"]}
            for idx in indices
        ]

        # Skip the API call if there's only one item
        if len(items) <= 1:
            refined_clusters.append(set(indices))
            continue

        groups = split_cluster_with_llm(items)
        for group in groups:
            if group:
                refined_clusters.append(set(group))

        print(f"  Cluster {ci}: {len(indices)} items -> {len(groups)} group(s)")

    # --- Pass 2: try to match each singleton to an existing cluster ---
    # Pool together the original singletons AND any 1-element groups Pass 1
    # produced when splitting clusters. Without this, an over-eager LLM split
    # would orphan valid duplicates with no path back into a cluster.
    multi_clusters = [c for c in refined_clusters if len(c) > 1]
    pass1_orphans = [c for c in refined_clusters if len(c) == 1]
    all_singletons = list(singletons) + pass1_orphans

    print(f"\nPass 2: matching {len(all_singletons)} singletons "
          f"({len(singletons)} original + {len(pass1_orphans)} from Pass 1 splits) "
          f"against existing clusters")
    remaining_singletons: list[set[int]] = []
    matched_count = 0

    for singleton_set in all_singletons:
        idx = next(iter(singleton_set))
        singleton_name = df.iloc[idx]["product_name"]
        singleton_cat = df.iloc[idx]["category"]

        # Restrict candidates to same-category clusters (cheap and safe)
        candidate_clusters = [
            c for c in multi_clusters
            if any(df.iloc[i]["category"] == singleton_cat for i in c)
        ]

        if not candidate_clusters:
            remaining_singletons.append(singleton_set)
            continue

        candidate_names = [
            [df.iloc[i]["product_name"] for i in c]
            for c in candidate_clusters
        ]

        match_idx = match_singleton_with_llm(singleton_name, candidate_names)
        if match_idx is not None:
            candidate_clusters[match_idx].add(idx)
            matched_count += 1
            print(f"  Matched: {singleton_name}")
        else:
            remaining_singletons.append(singleton_set)

    print(f"\nPass 2 result: matched {matched_count}/{len(all_singletons)} singletons")

    # --- Pass 3: merge clusters that refer to the same product ---
    # Pass 1 (split) and Pass 2 (singleton matching) can still leave the SAME
    # product in two separate clusters when both clusters were already multi-item
    # at the start of Pass 2 (e.g. a Hebrew-named XM5 cluster and an English-named
    # XM5 cluster). Pass 3 catches these.
    print(f"\nPass 3: checking {len(multi_clusters)} clusters for cross-cluster merges")
    merged_clusters, merge_count = merge_split_clusters(df, multi_clusters)
    print(f"Pass 3 result: {merge_count} merges -> {len(merged_clusters)} clusters")

    # Final = merged multi-clusters + whatever singletons couldn't find a home.
    final = list(merged_clusters) + remaining_singletons

    print(f"\n{'=' * 60}")
    print(f"FINAL: {len([c for c in final if len(c) > 1])} clusters + "
          f"{len([c for c in final if len(c) == 1])} singletons")
    print(f"{'=' * 60}")

    return final


if __name__ == "__main__":
    # Smoke test: run embedding stage then LLM refinement
    from embeddings import run_embedding_stage

    df, clusters, singletons, _, _ = run_embedding_stage()
    final = refine_clusters(df, clusters, singletons)

    print("\n--- Final clusters ---")
    multi = [c for c in final if len(c) > 1]
    singles = [c for c in final if len(c) == 1]
    for i, cluster in enumerate(sorted(multi, key=lambda c: -len(c)), 1):
        print(f"\nCluster {i} ({len(cluster)} items):")
        for idx in sorted(cluster):
            row = df.iloc[idx]
            print(f"  - [{row['product_id']}] {row['product_name']} (₪{row['price']})")
    if singles:
        print(f"\n--- {len(singles)} unmatched singletons ---")
        for s in singles:
            idx = next(iter(s))
            row = df.iloc[idx]
            print(f"  - [{row['product_id']}] {row['product_name']}")
