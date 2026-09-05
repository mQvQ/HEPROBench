from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def harrell_c_index(
    event_observed: Iterable[bool | int | float],
    event_time: Iterable[float],
    risk: Iterable[float],
    *,
    tied_tolerance: float = 1e-8,
) -> float:
    """Harrell's concordance index with higher scores indicating higher risk."""

    events = np.asarray(list(event_observed), dtype=bool)
    times = np.asarray(list(event_time), dtype=np.float64)
    risks = np.asarray(list(risk), dtype=np.float64)
    if not (len(events) == len(times) == len(risks)):
        raise ValueError("event_observed, event_time, and risk must have equal lengths")
    if len(events) < 2 or not np.isfinite(times).all() or not np.isfinite(risks).all():
        return float("nan")

    concordant = 0.0
    comparable = 0
    for left in range(len(times) - 1):
        for right in range(left + 1, len(times)):
            if times[left] == times[right]:
                # Two events at the same time do not establish an ordering. If
                # only one is censored, the observed event is known to happen no
                # later and the pair remains usable.
                if events[left] == events[right]:
                    continue
                earlier = left if events[left] else right
                later = right if earlier == left else left
            elif times[left] < times[right]:
                if not events[left]:
                    continue
                earlier, later = left, right
            else:
                if not events[right]:
                    continue
                earlier, later = right, left

            comparable += 1
            delta = risks[earlier] - risks[later]
            if abs(delta) <= tied_tolerance:
                concordant += 0.5
            elif delta > 0:
                concordant += 1.0
    return concordant / comparable if comparable else float("nan")


def aggregate_patient_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average slide risks/probabilities while retaining patient/fold labels."""

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["fold"]), str(row["patient_id"]))].append(row)
    output: list[dict[str, Any]] = []
    label_keys = (
        "event_time",
        "censor",
        "event_observed",
        "survival_bin",
        "label",
        "class_index",
    )
    for (fold, patient_id), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "fold": fold,
            "patient_id": patient_id,
            "slide_id": patient_id,
            "n_slides": len({str(item["slide_id"]) for item in group}),
        }
        for key in label_keys:
            values = [item.get(key) for item in group if item.get(key) not in (None, "")]
            if values:
                if any(str(item) != str(values[0]) for item in values[1:]):
                    raise ValueError(f"Inconsistent {key} within patient={patient_id}, fold={fold}")
                result[key] = values[0]
        if "risk" in group[0]:
            result["risk"] = float(np.mean([float(item["risk"]) for item in group]))
        probability_keys = sorted(key for key in group[0] if key.startswith("probability_"))
        for key in probability_keys:
            result[key] = float(np.mean([float(item[key]) for item in group]))
        output.append(result)
    return output


def survival_fold_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["fold"])].append(row)
    fold_rows: list[dict[str, Any]] = []
    for fold, group in sorted(by_fold.items()):
        value = harrell_c_index(
            (bool(int(float(item["event_observed"]))) for item in group),
            (float(item["event_time"]) for item in group),
            (float(item["risk"]) for item in group),
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_samples": len(group),
                "n_patients": len({str(item["patient_id"]) for item in group}),
                "n_events": sum(int(float(item["event_observed"])) for item in group),
                "c_index": value,
            }
        )
    values = np.asarray([float(row["c_index"]) for row in fold_rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    summary = {
        "n_folds": len(fold_rows),
        "mean_c_index": float(np.mean(finite)) if len(finite) else float("nan"),
        "std_c_index": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else float("nan"),
    }
    return fold_rows, summary


def patient_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap the mean fold C-index by resampling patients within fold."""

    if iterations <= 0:
        return {"enabled": False, "iterations_requested": iterations}
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_patient_predictions(rows):
        by_fold[int(row["fold"])].append(row)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        fold_values: list[float] = []
        for group in by_fold.values():
            indices = rng.integers(0, len(group), len(group))
            sampled = [group[int(index)] for index in indices]
            value = harrell_c_index(
                (bool(int(float(item["event_observed"]))) for item in sampled),
                (float(item["event_time"]) for item in sampled),
                (float(item["risk"]) for item in sampled),
            )
            if math.isfinite(value):
                fold_values.append(value)
        if fold_values:
            values.append(float(np.mean(fold_values)))
    alpha = (1.0 - confidence_level) / 2.0
    array = np.asarray(values, dtype=np.float64)
    return {
        "enabled": True,
        "method": "patient_cluster_within_fold",
        "seed": seed,
        "confidence_level": confidence_level,
        "iterations_requested": iterations,
        "iterations_valid": len(values),
        "bootstrap_mean_c_index": float(np.mean(array)) if len(array) else float("nan"),
        "ci_lower": float(np.quantile(array, alpha)) if len(array) else float("nan"),
        "ci_upper": float(np.quantile(array, 1.0 - alpha)) if len(array) else float("nan"),
    }


def logrank_test(rows: list[dict[str, Any]], group_key: str = "risk_group") -> dict[str, float]:
    """Two-group Mantel-Cox log-rank test without an optional lifelines import."""

    groups = sorted({str(row[group_key]) for row in rows})
    if len(groups) != 2:
        return {"chi_square": float("nan"), "p_value": float("nan")}
    event_times = sorted(
        {
            float(row["event_time"])
            for row in rows
            if bool(int(float(row["event_observed"])))
        }
    )
    observed = 0.0
    expected = 0.0
    variance = 0.0
    first = groups[0]
    for time in event_times:
        at_risk = [row for row in rows if float(row["event_time"]) >= time]
        events = [
            row
            for row in rows
            if float(row["event_time"]) == time and bool(int(float(row["event_observed"])))
        ]
        n_total = len(at_risk)
        d_total = len(events)
        if n_total <= 1 or d_total == 0:
            continue
        n_first = sum(str(row[group_key]) == first for row in at_risk)
        d_first = sum(str(row[group_key]) == first for row in events)
        observed += d_first
        expected += d_total * n_first / n_total
        variance += (
            n_first
            * (n_total - n_first)
            * d_total
            * (n_total - d_total)
            / (n_total * n_total * (n_total - 1))
        )
    statistic = (observed - expected) ** 2 / variance if variance > 0 else float("nan")
    try:
        from scipy.stats import chi2

        p_value = float(chi2.sf(statistic, 1)) if math.isfinite(statistic) else float("nan")
    except ImportError:
        p_value = math.erfc(math.sqrt(statistic / 2.0)) if math.isfinite(statistic) else float("nan")
    return {"chi_square": statistic, "p_value": p_value}


def kaplan_meier_rows(rows: list[dict[str, Any]], group_key: str = "risk_group") -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in sorted({str(row[group_key]) for row in rows}):
        selected = [row for row in rows if str(row[group_key]) == group]
        survival = 1.0
        output.append(
            {
                "risk_group": group,
                "time": 0.0,
                "survival_probability": survival,
                "n_at_risk": len(selected),
                "n_events": 0,
            }
        )
        for time in sorted({float(row["event_time"]) for row in selected}):
            at_risk = sum(float(row["event_time"]) >= time for row in selected)
            events = sum(
                float(row["event_time"]) == time and bool(int(float(row["event_observed"])))
                for row in selected
            )
            if events:
                survival *= 1.0 - events / at_risk
            output.append(
                {
                    "risk_group": group,
                    "time": time,
                    "survival_probability": survival,
                    "n_at_risk": at_risk,
                    "n_events": events,
                }
            )
    return output


def assign_median_risk_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign fold-specific median groups so fold risk scales do not leak across folds."""

    output = [dict(row) for row in rows]
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_fold[int(row["fold"])].append(row)
    for group in by_fold.values():
        threshold = float(np.median([float(row["risk"]) for row in group]))
        for row in group:
            row["risk_group"] = "high" if float(row["risk"]) >= threshold else "low"
            row["risk_threshold"] = threshold
    return output
