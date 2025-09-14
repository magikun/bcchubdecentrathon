from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def pct_improvement(baseline: float, new: float, lower_is_better: bool = True) -> float:
    if baseline is None or new is None:
        return None
    if baseline == 0:
        return 100.0 if new < baseline and lower_is_better else 0.0
    delta = baseline - new if lower_is_better else new - baseline
    return 100.0 * (delta / abs(baseline))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))

    # Aggregate metrics
    n = 0
    sums = {"cer": 0.0, "wer": 0.0, "norm_lev": 0.0}
    sums_base = {"cer": 0.0, "wer": 0.0, "norm_lev": 0.0}
    em_sum = 0.0
    json_valid_count = 0
    present_top_keys_counts: Dict[str, int] = {}
    # Field-level aggregates
    field_agg_scores: Dict[str, Dict[str, float]] = {}
    field_counts: Dict[str, int] = {}
    # noisy subset
    n_noisy = 0
    sums_noisy = {"cer": 0.0, "wer": 0.0, "norm_lev": 0.0}
    sums_base_noisy = {"cer": 0.0, "wer": 0.0, "norm_lev": 0.0}
    NOISE_THRESH = 0.5
    for row in data:
        m = row.get("metrics") or {}
        t = row.get("tesseract") or {}
        if m.get("cer") is not None:
            n += 1
            for k in sums:
                sums[k] += float(m.get(k) or 0.0)
                sums_base[k] += float((t.get(k) if t else 0.0) or 0.0)
            if (row.get("noise_score") or 0.0) >= NOISE_THRESH:
                n_noisy += 1
                for k in sums_noisy:
                    sums_noisy[k] += float(m.get(k) or 0.0)
                    sums_base_noisy[k] += float((t.get(k) if t else 0.0) or 0.0)
        if row.get("exact_match") is not None:
            em_sum += float(row.get("exact_match") or 0.0)
        if row.get("json_valid"):
            json_valid_count += 1
        for k in (row.get("present_top_keys") or []):
            present_top_keys_counts[k] = present_top_keys_counts.get(k, 0) + 1

        # Per-field aggregation
        fs = row.get("field_scores") or {}
        for field_path, scores in fs.items():
            agg = field_agg_scores.setdefault(field_path, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
            field_counts[field_path] = field_counts.get(field_path, 0) + 1
            for metric_name in ["accuracy", "precision", "recall", "f1"]:
                agg[metric_name] += float(scores.get(metric_name) or 0.0)

    out = {}
    if n > 0:
        avg = {k: v / n for k, v in sums.items()}
        avg_base = {k: v / n for k, v in sums_base.items()}
        out["avg_metrics"] = avg
        out["avg_tesseract"] = avg_base
        out["improvements_percent"] = {
            k: pct_improvement(avg_base[k], avg[k], lower_is_better=True) for k in avg
        }
        out["exact_match_rate"] = em_sum / n if n > 0 else None
        out["json_valid_rate"] = json_valid_count / n
        out["top_key_presence_rate"] = {k: c / n for k, c in present_top_keys_counts.items()}

        # Average per-field metrics
        out["field_level_avg"] = {
            fp: {m: v / field_counts[fp] for m, v in agg.items()} for fp, agg in field_agg_scores.items()
        }
    if n_noisy > 0:
        avg_noisy = {k: v / n_noisy for k, v in sums_noisy.items()}
        avg_base_noisy = {k: v / n_noisy for k, v in sums_base_noisy.items()}
        out["noisy_threshold"] = 0.5
        out["avg_metrics_noisy"] = avg_noisy
        out["avg_tesseract_noisy"] = avg_base_noisy
        out["improvements_percent_noisy"] = {
            k: pct_improvement(avg_base_noisy[k], avg_noisy[k], lower_is_better=True) for k in avg_noisy
        }

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


