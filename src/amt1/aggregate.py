# SPDX-License-Identifier: Apache-2.0
"""Aggregate category point files using the released benchmark weights."""
from pathlib import Path

import numpy as np

from .io import read_json, write_json

METRICS = ("image_auroc", "image_ap", "image_f1", "pixel_auroc", "pixel_ap",
           "pixel_f1", "aupro", "pro_at_fpr_0p01")


def aggregate(manifest_root: str | Path, metric_root: str | Path,
              output: str | Path) -> dict:
    """Aggregate admitted categories with explicit NA and coverage handling.

    Args:
        manifest_root: Directory containing portable per-category manifests.
        metric_root: Directory recursively containing category evaluation JSON files.
        output: Destination for category, dataset, and equal-dataset domain means.

    Returns:
        Complete category, dataset and domain aggregates in fraction units.

    Raises:
        ValueError: A category is absent, duplicated, or has incompatible counts.
    """
    manifests = [read_json(path) for path in sorted(Path(manifest_root).glob("*.json"))]
    measured = {}
    for path in Path(metric_root).rglob("*.json"):
        record = read_json(path)
        if "key" not in record or "metrics" not in record:
            continue
        key = record["key"]
        if key in measured:
            raise ValueError(f"Multiple point results for {key}.")
        measured[key] = record
    categories = {}
    datasets = {}
    dataset_domains = {}
    for manifest in manifests:
        key = manifest["key"]
        if key not in measured or measured[key]["count"] != manifest["count"]:
            raise ValueError(f"Missing or incomplete category result for {key}.")
        metrics = measured[key]["metrics"]
        if set(metrics) != set(METRICS):
            raise ValueError(f"Eight metric fields are required for {key}.")
        categories[key] = metrics
        dataset = manifest["dataset"]
        datasets.setdefault(dataset, []).append(metrics)
        domain = "medical_classification" if manifest["geometry"] == "classification-only" else (
            "medical_localization" if manifest["states"] == "clinical" else "industrial")
        dataset_domains[dataset] = domain
    dataset_means = {}
    for dataset, rows in datasets.items():
        dataset_means[dataset] = {}
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            dataset_means[dataset][metric] = float(np.mean(values)) if values else None
    domains = {}
    for domain in sorted(set(dataset_domains.values())):
        group = [dataset_means[name] for name in datasets if dataset_domains[name] == domain]
        domains[domain] = {}
        for metric in METRICS:
            values = [row[metric] for row in group if row[metric] is not None]
            domains[domain][metric] = float(np.mean(values)) if values else None
    result = dict(status="completed", count=len(manifests), categories=categories,
                  datasets=dataset_means, domains=domains,
                  weighting="Equal category within dataset, then equal dataset within domain")
    write_json(output, result)
    return result
