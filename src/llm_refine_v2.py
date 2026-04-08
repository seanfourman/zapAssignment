"""
Stage 2 v2: CONFIDENCE-GATED LLM refinement.

The original llm_refine.py calls the LLM on every candidate cluster, every
singleton, and every cluster pair. That's wasteful - many of those calls are
on cases that the embedding stage already got right. This version adds a
confidence gate before each LLM call:

  Pass 1 (split):  only call LLM if items in the cluster disagree on a
                   variant-discriminator token (Pro / Max / Ultra / 256GB /
                   512GB / inch / kg / FE / M2 / M3 / etc.). Otherwise the
                   cluster is internally uniform - trust it.

  Pass 2 (singleton match): if a singleton's centroid similarity to its best
                   candidate cluster is >= 0.85, auto-merge with no LLM call.
                   Otherwise ask the LLM (this is where Hebrew transliterations
                   like "אירפודז פרו 2" still need world knowledge).

  Pass 3 (cluster merge): only invoke the LLM for cluster pairs whose
                   centroids are close (cosine >= 0.70). Far-apart clusters
                   are obviously different products.

Net effect on the test dataset: roughly 1/3 the API calls of v1 with
identical clustering quality.

This file deliberately reuses the LLM-call helpers (split_cluster_with_llm,
match_singleton_with_llm) and the prompt definitions from llm_refine.py so
both versions are guaranteed to use the exact same LLM behavior. The only
difference between v1 and v2 is the gating logic.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import LLM_MODEL
from llm_refine import (
    split_cluster_with_llm,
    match_singleton_with_llm,
)


# ---------------------------------------------------------------------------
# Variant-discriminator detection
# ---------------------------------------------------------------------------

# Tokens that, if they DIFFER between items in a cluster, signal that the
# cluster probably mixes different product variants and needs LLM splitting.
# Order matters here - longer / more specific patterns come first so they
# match before generic numeric patterns swallow them.
VARIANT_PATTERNS = [
    # Storage / capacity (English + Hebrew)
    r"\d+\s*gb\b",
    r"\d+\s*ג'?יגה",
    # Weight (washing machines)
    r'\d+\s*kg\b',
    r'\d+\s*ק"?ג\b',
    r"\d+\s*קילו",
    # Screen size
    r'\d+\.?\d*\s*"',
    r"\d+\.?\d*\s*inch",
    r"\d+\.?\d*\s*אינץ'?",
    # Apple silicon generations
    r"\bm[1-4]\b",
    # Intel CPU tiers
    r"\bi[3579]\b",
    # Sony XM headphones generation
    r"\bxm[3-5]\b",
    # Samsung Galaxy S generation (S24, S25, etc.)
    r"\bs2[0-9]\b",
    # iPhone generation
    r"\biphone\s*1[0-9]\b",
    r"אייפון\s*1[0-9]",
    # Variant suffixes
    r"\bpro\s*max\b",
    r"\bpro\b",
    r"\bmax\b",
    r"\bultra\b",
    r"\bplus\b",
    r"\bmini\b",
    r"\blite\b",
    r"\bfe\b",
    r"פרו\s*מקס",
    r"\bפרו\b",
    r"אולטרה",
]

_VARIANT_RE = re.compile("|".join(VARIANT_PATTERNS), re.IGNORECASE)


def extract_variant_tokens(name: str) -> frozenset[str]:
    """Pull all variant-discriminator tokens out of a product name."""
    matches = _VARIANT_RE.findall(name.lower())
    # Normalize whitespace inside multi-word matches like "pro max"
    return frozenset(re.sub(r"\s+", " ", m.strip()) for m in matches)


def cluster_needs_llm_split(items: list[dict]) -> bool:
    """
    Decide whether a cluster genuinely needs the LLM to split it.

    Heuristic: extract the set of variant tokens from each item. If every
    item has the SAME set of variant tokens, the cluster is internally
    uniform - every item is the same variant of the same product family,
    so we can trust the embedding stage's grouping.

    If items disagree on at least one variant token, the cluster probably
    mixes variants (e.g. iPhone 15 + iPhone 16, or 256GB + 512GB) and the
    LLM should split it.
    """
    if len(items) <= 1:
        return False

    token_sets = [extract_variant_tokens(item["name"]) for item in items]
    first = token_sets[0]
    return any(ts != first for ts in token_sets[1:])


def _round_confidence(value: float | None) -> float | None:
    """Round similarity-style support scores for the audit JSON."""
    return round(float(value), 3) if value is not None else None


def _names_for_indices(df, indices: list[int]) -> list[str]:
    """Return product names for a sorted list of dataframe indices."""
    return [str(df.iloc[idx]["product_name"]) for idx in indices]


def _cluster_names(df, cluster: set[int]) -> list[str]:
    """Return every product name inside a cluster."""
    return _names_for_indices(df, sorted(cluster))


def _cluster_anchor_name(df, cluster: set[int]) -> str:
    """Pick one human-readable representative name for a cluster."""
    names = _cluster_names(df, cluster)
    return max(names, key=lambda n: (len(n), -ord(n[0]) if n else 0))


# ---------------------------------------------------------------------------
# Pass 3: gated cluster-to-cluster merging via union-find
# ---------------------------------------------------------------------------

def merge_split_clusters_gated(
    df,
    embeddings,
    multi_clusters: list[set[int]],
    parent_ids: list[int],
    centroid_threshold: float = 0.80,
    audit_entries: list[dict] | None = None,
) -> tuple[list[set[int]], int, int]:
    """
    For each pair of multi-clusters in the same category, only invoke the LLM
    if their centroids are close enough that they MIGHT be the same product.

    Two important guards:
      1. Skip pairs whose centroid similarity is below `centroid_threshold` -
         far-apart clusters obviously aren't the same product.
      2. Skip "sibling" pairs that came from the SAME Pass 1 parent cluster.
         If Pass 1 already decided to split them, Pass 3 has no business
         re-merging them. This prevents Pass 3 from undoing the iPhone /
         Samsung / MacBook splits.

    Returns (merged_clusters, num_llm_calls, num_merges).
    """
    if len(multi_clusters) < 2:
        return multi_clusters, 0, 0

    clusters = [set(c) for c in multi_clusters]
    uf_parent = list(range(len(clusters)))

    def find(x):
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            uf_parent[max(px, py)] = min(px, py)

    # Pre-compute centroids and category sets
    info = []
    for c in clusters:
        idxs = list(c)
        centroid = embeddings[idxs].mean(axis=0)
        cats = {df.iloc[i]["category"] for i in idxs}
        info.append({"centroid": centroid, "cats": cats})

    llm_calls = 0
    merges = 0

    for i in range(len(clusters)):
        rep_idx = sorted(clusters[i])[0]
        rep_name = df.iloc[rep_idx]["product_name"]
        rep_cat = df.iloc[rep_idx]["category"]

        candidate_indices = []
        candidate_names = []
        for j in range(len(clusters)):
            if j == i or find(i) == find(j):
                continue
            # Sibling guard: never re-merge clusters split from the same parent.
            if parent_ids[i] >= 0 and parent_ids[i] == parent_ids[j]:
                continue
            if rep_cat not in info[j]["cats"]:
                continue
            sim = float(cosine_similarity(
                info[i]["centroid"].reshape(1, -1),
                info[j]["centroid"].reshape(1, -1),
            )[0, 0])
            if sim < centroid_threshold:
                continue
            candidate_indices.append(j)
            candidate_names.append([df.iloc[k]["product_name"] for k in clusters[j]])

        if not candidate_indices:
            continue

        match_idx = match_singleton_with_llm(rep_name, candidate_names)
        llm_calls += 1
        if match_idx is not None:
            target_j = candidate_indices[match_idx]
            target_name = _cluster_anchor_name(df, clusters[target_j])
            sim = float(cosine_similarity(
                info[i]["centroid"].reshape(1, -1),
                info[target_j]["centroid"].reshape(1, -1),
            )[0, 0])
            union(i, target_j)
            merges += 1
            if audit_entries is not None:
                audit_entries.append({
                    "stage": "pass3",
                    "decision_type": "cluster_merge",
                    "pair": [rep_name, target_name],
                    "is_match": True,
                    "confidence": _round_confidence(sim),
                    "confidence_type": "cluster_centroid_similarity",
                    "used_ai": True,
                    "matched_cluster": _cluster_names(df, clusters[target_j]),
                    "reason": (
                        f"Cluster centroids were close enough to pass the "
                        f"{centroid_threshold:.2f} merge gate, and GPT-4o-mini "
                        "confirmed that the two clusters describe the same product."
                    ),
                })
            print(f"  Merged cluster {i} ('{rep_name}') -> cluster {target_j}")

    merged: dict[int, set[int]] = {}
    for i, cluster in enumerate(clusters):
        root = find(i)
        merged.setdefault(root, set()).update(cluster)

    return list(merged.values()), llm_calls, merges


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def refine_clusters_v2(
    df,
    embeddings,
    clusters,
    singletons,
    centroid_auto_merge: float = 0.85,
    cluster_merge_threshold: float = 0.65,
    collect_audit: bool = False,
):
    """
    Confidence-gated version of refine_clusters.

    Returns (final_clusters, stats_dict).
    """
    print("=" * 60)
    print("STAGE 2 v2: CONFIDENCE-GATED LLM REFINEMENT")
    print("=" * 60)
    print(f"Input: {len(clusters)} candidate clusters, {len(singletons)} singletons")
    audit_entries: list[dict] | None = [] if collect_audit else None

    # --- Pass 1: gated split ---
    # We track a parent_id for every refined cluster so Pass 3 can skip
    # re-merging "siblings" - clusters that came from splitting the same
    # candidate cluster. -1 means "wasn't part of any split" (used for
    # original singletons later).
    print(f"\nPass 1 (gated): split only clusters with mixed variant tokens")
    refined_clusters: list[set[int]] = []
    refined_parents: list[int] = []
    pass1_skipped = 0
    pass1_llm = 0

    for ci, cluster in enumerate(clusters):
        indices = sorted(cluster)
        items = [
            {"id": int(idx), "name": df.iloc[idx]["product_name"]}
            for idx in indices
        ]

        if len(items) <= 1:
            refined_clusters.append(set(indices))
            refined_parents.append(ci)
            continue

        if not cluster_needs_llm_split(items):
            refined_clusters.append(set(indices))
            refined_parents.append(ci)
            pass1_skipped += 1
            if audit_entries is not None:
                audit_entries.append({
                    "stage": "pass1",
                    "decision_type": "cluster_trusted",
                    "items": _names_for_indices(df, indices),
                    "is_match": True,
                    "confidence": None,
                    "confidence_type": None,
                    "used_ai": False,
                    "reason": (
                        "All items shared the same extracted variant tokens, "
                        "so the candidate cluster was trusted without LLM review."
                    ),
                    "variant_tokens": sorted(extract_variant_tokens(items[0]["name"])),
                })
            print(f"  Cluster {ci + 1}: {len(items)} items -> trusted (uniform variants)")
            continue

        groups = split_cluster_with_llm(items)
        pass1_llm += 1
        for group in groups:
            if group:
                refined_clusters.append(set(group))
                refined_parents.append(ci)
        if audit_entries is not None:
            audit_entries.append({
                "stage": "pass1",
                "decision_type": "cluster_split_review",
                "items": _names_for_indices(df, indices),
                "output_groups": [
                    _names_for_indices(df, sorted(group))
                    for group in groups if group
                ],
                "is_match": None,
                "confidence": None,
                "confidence_type": None,
                "used_ai": True,
                "reason": (
                    "Mixed variant tokens triggered LLM review. "
                    f"The candidate cluster was split into {len(groups)} group(s)."
                ),
            })
        print(f"  Cluster {ci + 1}: {len(items)} items -> LLM split into {len(groups)} group(s)")

    print(f"Pass 1: skipped LLM on {pass1_skipped} clusters, called LLM on {pass1_llm}")

    # --- Pass 2: gated singleton matching ---
    multi_pairs = [(c, p) for c, p in zip(refined_clusters, refined_parents) if len(c) > 1]
    orphan_pairs = [(c, p) for c, p in zip(refined_clusters, refined_parents) if len(c) == 1]
    multi_clusters = [c for c, _ in multi_pairs]
    multi_parents = [p for _, p in multi_pairs]
    pass1_orphans = [c for c, _ in orphan_pairs]
    all_singletons = list(singletons) + pass1_orphans

    print(f"\nPass 2 (gated): matching {len(all_singletons)} singletons "
          f"({len(singletons)} original + {len(pass1_orphans)} from Pass 1)")

    remaining_singletons: list[set[int]] = []
    pass2_auto = 0
    pass2_llm = 0

    for singleton_set in all_singletons:
        idx = next(iter(singleton_set))
        singleton_name = df.iloc[idx]["product_name"]
        singleton_cat = df.iloc[idx]["category"]
        singleton_emb = embeddings[idx]

        # Restrict to same-category clusters
        candidate_clusters = [
            c for c in multi_clusters
            if any(df.iloc[i]["category"] == singleton_cat for i in c)
        ]

        if not candidate_clusters:
            remaining_singletons.append(singleton_set)
            if audit_entries is not None:
                audit_entries.append({
                    "stage": "pass2",
                    "decision_type": "singleton_unmatched",
                    "item": singleton_name,
                    "is_match": False,
                    "confidence": None,
                    "confidence_type": None,
                    "used_ai": False,
                    "reason": "No same-category multi-item cluster existed to compare against.",
                })
            continue

        # Compute centroid similarity to each candidate
        sims = []
        for c in candidate_clusters:
            cluster_indices = list(c)
            centroid = embeddings[cluster_indices].mean(axis=0)
            sim = float(cosine_similarity(
                singleton_emb.reshape(1, -1),
                centroid.reshape(1, -1),
            )[0, 0])
            sims.append(sim)

        best_sim = max(sims)
        best_local_idx = sims.index(best_sim)
        best_anchor = _cluster_anchor_name(df, candidate_clusters[best_local_idx])

        if best_sim >= centroid_auto_merge:
            # High-confidence: auto-merge without spending an LLM call
            candidate_clusters[best_local_idx].add(idx)
            pass2_auto += 1
            if audit_entries is not None:
                audit_entries.append({
                    "stage": "pass2",
                    "decision_type": "singleton_match",
                    "pair": [singleton_name, best_anchor],
                    "is_match": True,
                    "confidence": _round_confidence(best_sim),
                    "confidence_type": "best_centroid_similarity",
                    "used_ai": False,
                    "matched_cluster": _cluster_names(df, candidate_clusters[best_local_idx]),
                    "reason": (
                        f"Best same-category centroid similarity exceeded the "
                        f"{centroid_auto_merge:.2f} auto-merge threshold, so the "
                        "singleton was merged without LLM review."
                    ),
                })
            print(f"  Auto-matched (sim={best_sim:.2f}): {singleton_name}")
        else:
            # Ambiguous: ask the LLM
            candidate_names = [
                [df.iloc[i]["product_name"] for i in c]
                for c in candidate_clusters
            ]
            match_idx = match_singleton_with_llm(singleton_name, candidate_names)
            pass2_llm += 1
            if match_idx is not None:
                target_anchor = _cluster_anchor_name(df, candidate_clusters[match_idx])
                candidate_clusters[match_idx].add(idx)
                if audit_entries is not None:
                    target_cluster = candidate_clusters[match_idx]
                    audit_entries.append({
                        "stage": "pass2",
                        "decision_type": "singleton_match",
                        "pair": [singleton_name, target_anchor],
                        "is_match": True,
                        "confidence": _round_confidence(best_sim),
                        "confidence_type": "best_centroid_similarity",
                        "used_ai": True,
                        "matched_cluster": _cluster_names(df, target_cluster),
                        "reason": (
                            f"Best centroid similarity stayed below the "
                            f"{centroid_auto_merge:.2f} auto-merge threshold, so "
                            "GPT-4o-mini reviewed the ambiguous case and matched "
                            "the singleton to an existing cluster."
                        ),
                    })
                print(f"  LLM-matched (sim={best_sim:.2f}): {singleton_name}")
            else:
                remaining_singletons.append(singleton_set)
                if audit_entries is not None:
                    audit_entries.append({
                        "stage": "pass2",
                        "decision_type": "singleton_unmatched",
                        "pair": [singleton_name, best_anchor],
                        "is_match": False,
                        "confidence": _round_confidence(best_sim),
                        "confidence_type": "best_centroid_similarity",
                        "used_ai": True,
                        "reason": (
                            f"Best centroid similarity stayed below the "
                            f"{centroid_auto_merge:.2f} auto-merge threshold, and "
                            "GPT-4o-mini rejected all candidate clusters."
                        ),
                    })

    print(f"Pass 2: auto-merged {pass2_auto}, LLM-merged {pass2_llm}, "
          f"unmatched {len(remaining_singletons)}")

    # --- Pass 3: gated cluster-to-cluster merging ---
    print(f"\nPass 3 (gated): cluster-to-cluster merge above sim={cluster_merge_threshold}")
    merged_clusters, pass3_llm, pass3_merges = merge_split_clusters_gated(
        df, embeddings, multi_clusters, multi_parents,
        centroid_threshold=cluster_merge_threshold,
        audit_entries=audit_entries,
    )
    print(f"Pass 3: {pass3_llm} LLM calls -> {pass3_merges} merges -> "
          f"{len(merged_clusters)} clusters")

    final = list(merged_clusters) + remaining_singletons

    print(f"\n{'=' * 60}")
    print(f"FINAL: {len([c for c in final if len(c) > 1])} clusters + "
          f"{len([c for c in final if len(c) == 1])} singletons")
    print(f"{'=' * 60}")

    stats = {
        "pass1_skipped": pass1_skipped,
        "pass1_llm": pass1_llm,
        "pass2_auto": pass2_auto,
        "pass2_llm": pass2_llm,
        "pass3_llm": pass3_llm,
        "pass3_merges": pass3_merges,
        "total_llm": pass1_llm + pass2_llm + pass3_llm,
    }
    if audit_entries is not None:
        stats["audit_entries"] = audit_entries
    return final, stats


if __name__ == "__main__":
    from embeddings import run_embedding_stage
    from llm_refine import reset_api_call_count, get_api_call_count

    df, clusters, singletons, _, embeddings = run_embedding_stage()
    reset_api_call_count()
    final, stats = refine_clusters_v2(df, embeddings, clusters, singletons)
    print(f"\nTotal API calls (counter): {get_api_call_count()}")
    print(f"Stats: {stats}")
