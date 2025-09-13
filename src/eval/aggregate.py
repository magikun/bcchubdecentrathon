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
    if n_noisy > 0:
        avg_noisy = {k: v / n_noisy for k, v in sums_noisy.items()}
        avg_base_noisy = {k: v / n_noisy for k, v in sums_base_noisy.items()}
        out["noisy_threshold"] = 0.5
        out["avg_metrics_noisy"] = avg_noisy
        out["avg_tesseract_noisy"] = avg_base_noisy
        out["improvements_percent_noisy"] = {
            k: pct_improvement(avg_base_noisy[k], avg_noisy[k], lower_is_better=True) for k in avg_noisy
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


