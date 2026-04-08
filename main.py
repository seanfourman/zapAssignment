"""
End-to-end pipeline entry point.

Wires the three stages together in the order they need to run:

  Stage 1 - embeddings.run_embedding_stage
            Reads data/products.csv, generates multilingual embeddings,
            does pairwise + centroid clustering. Output: candidate
            clusters + leftover singletons.

  Stage 2 - llm_refine_v2.refine_clusters_v2
            Confidence-gated LLM refinement. Splits over-merged clusters,
            matches singletons, merges Hebrew/English siblings. Output:
            final clusters of true duplicates.

            (The original always-LLM v1 lives in llm_refine.py for
             comparison - see data/compare_approaches.py.)

  Stage 3 - deduplicator.consolidate
            Picks the canonical name and lowest price per cluster,
            writes the unified catalog to data/deduplicated_products.csv.
            Also writes a companion explainability JSON next to the CSV.

Run with:
    python main.py                      # uses default INPUT_CSV / OUTPUT_CSV
    python main.py --input my.csv       # override input
    python main.py --output out.csv     # override output
    python main.py --v1                 # use the always-LLM v1 instead of v2
"""

import argparse
import sys
import os

# Make sure Hebrew paths / product names don't crash on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Make src/ importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import INPUT_CSV, OUTPUT_CSV
from audit import derive_audit_path, build_audit_report, write_audit_report
from embeddings import run_embedding_stage
from llm_refine import refine_clusters as refine_v1
from llm_refine import reset_api_call_count, get_api_call_count
from llm_refine_v2 import refine_clusters_v2
from deduplicator import consolidate, print_summary


def main():
    parser = argparse.ArgumentParser(
        description="Unify duplicate product listings and surface the lowest price."
    )
    parser.add_argument("--input",  default=INPUT_CSV,
                        help=f"Input CSV path (default: {INPUT_CSV})")
    parser.add_argument("--output", default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    parser.add_argument("--v1", action="store_true",
                        help="Use always-LLM v1 instead of confidence-gated v2")
    args = parser.parse_args()

    # ----- Stage 1: embeddings -----
    df, clusters, singletons, _, embeddings = run_embedding_stage(args.input)

    # ----- Stage 2: LLM refinement -----
    reset_api_call_count()
    stage2_stats = None
    if args.v1:
        final_clusters = refine_v1(df, clusters, singletons)
    else:
        final_clusters, stage2_stats = refine_clusters_v2(
            df, embeddings, clusters, singletons, collect_audit=True
        )
    api_calls = get_api_call_count()

    # ----- Stage 3: price consolidation -----
    unified = consolidate(df, final_clusters)
    print_summary(unified)

    unified.to_csv(args.output, index=False, encoding="utf-8-sig")
    audit_output = derive_audit_path(args.output)

    if args.v1:
        audit_report = build_audit_report(
            pipeline_version="v1",
            input_csv=args.input,
            output_csv=args.output,
            n_input_rows=len(df),
            final_clusters=final_clusters,
            api_calls=api_calls,
            note=(
                "Detailed decision-level audit is currently produced by the default "
                "v2 pipeline. This file only stores the high-level run summary."
            ),
        )
    else:
        audit_entries = stage2_stats.pop("audit_entries", []) if stage2_stats else []
        audit_report = build_audit_report(
            pipeline_version="v2",
            input_csv=args.input,
            output_csv=args.output,
            n_input_rows=len(df),
            final_clusters=final_clusters,
            api_calls=api_calls,
            stage2_stats=stage2_stats,
            decisions=audit_entries,
        )

    write_audit_report(audit_output, audit_report)

    print(f"\nLLM API calls used: {api_calls}")
    print(f"Saved unified catalog: {args.output}")
    print(f"Saved explainability audit: {audit_output}")


if __name__ == "__main__":
    main()
