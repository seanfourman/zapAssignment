"""
Helpers for writing a human-readable explainability artifact alongside the
deduplicated CSV output.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def derive_audit_path(output_csv: str) -> str:
    """Place the audit report next to the CSV as <stem>.audit.json."""
    root, _ = os.path.splitext(output_csv)
    return f"{root}.audit.json"


def _sanitize_path(path: str) -> str:
    """
    Store repo-relative paths in the audit JSON so local machine details do not
    leak into a submission artifact.
    """
    abs_path = os.path.abspath(path)
    try:
        if os.path.commonpath([_PROJECT_ROOT, abs_path]) == _PROJECT_ROOT:
            rel = os.path.relpath(abs_path, _PROJECT_ROOT)
            return rel.replace("\\", "/")
    except ValueError:
        # Different drives on Windows can make commonpath fail; fall back below.
        pass
    return os.path.basename(path)


def build_audit_report(
    *,
    pipeline_version: str,
    input_csv: str,
    output_csv: str,
    n_input_rows: int,
    final_clusters: list[set[int]],
    api_calls: int,
    stage2_stats: dict[str, Any] | None = None,
    decisions: list[dict[str, Any]] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Build the JSON structure written to disk."""
    report: dict[str, Any] = {
        "pipeline_version": pipeline_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": _sanitize_path(input_csv),
        "output_csv": _sanitize_path(output_csv),
        "summary": {
            "input_listings": n_input_rows,
            "unified_products": len(final_clusters),
            "multi_item_clusters": sum(1 for c in final_clusters if len(c) > 1),
            "singletons": sum(1 for c in final_clusters if len(c) == 1),
            "llm_api_calls": api_calls,
            "decision_entries": len(decisions or []),
        },
        "confidence_note": (
            "The `confidence` field is a heuristic similarity signal when available "
            "(for example centroid similarity), not a calibrated probability."
        ),
    }

    if stage2_stats:
        report["stage2_stats"] = stage2_stats
    if decisions:
        report["decisions"] = decisions
    if note:
        report["note"] = note

    return report


def write_audit_report(path: str, report: dict[str, Any]) -> None:
    """Write the audit report as pretty-printed UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
