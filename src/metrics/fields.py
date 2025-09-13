from __future__ import annotations

from typing import Dict, Tuple


def field_level_scores(gt: Dict, pred: Dict) -> Dict[str, Dict[str, float]]:
    """Compute per-field accuracy, precision, recall, F1.

    Assumes both are flat or nested JSON with scalar leaves. Flattens by dotted paths.
    """
    def flatten(prefix: str, obj: Dict, out: Dict):
        for k, v in (obj or {}).items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten(path, v, out)
            else:
                out[path] = v

    gt_flat, pr_flat = {}, {}
    flatten("", gt, gt_flat)
    flatten("", pred, pr_flat)

    keys = sorted(set(gt_flat.keys()) | set(pr_flat.keys()))
    result: Dict[str, Dict[str, float]] = {}
    for k in keys:
        gt_has = k in gt_flat and gt_flat[k] is not None and gt_flat[k] != ""
        pr_has = k in pr_flat and pr_flat[k] is not None and pr_flat[k] != ""
        correct = gt_has and pr_has and str(gt_flat[k]).strip() == str(pr_flat[k]).strip()

        tp = 1.0 if correct else 0.0
        fp = 1.0 if (pr_has and not correct) else 0.0
        fn = 1.0 if (gt_has and not correct) else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = 1.0 if correct else 0.0
        result[k] = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    return result


def exact_match(gt: Dict, pred: Dict) -> float:
    return 1.0 if gt == pred else 0.0


