"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }

def dubbing_scorecard(
    # calculates an overall score based on 5 criterias
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:

    if not metrics or not aligned:
        return {
            "timing_accuracy": 0.0,
            "stretch_quality": 0.0,
            "drift_score": 0.0,
            "naturalness": 0.0,
            "retry_rate": 0.0,
            "overall": 0.0,
        }

    report = align_report or clip_evaluation_report(metrics, aligned)

    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    timing_accuracy = max(0.0, 1.0 - mean_err / 3.0)

    pctsevere = report.get("pct_severe_stretch", 0.0)
    stretch_quality = max(0.0, 1.0 - pctsevere / 100.0)

    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    drift_score = max(0.0, 1.0 - drift / 5.0)



    speaking_rates = []
    for m in metrics:
        if m.source_duration_s > 0:
            rate = m.tgt_char_count / m.source_duration_s
            speaking_rates.append(rate)



    if len(speaking_rates) >= 2:
        mean_r = _stats.mean(speaking_rates)
        stdev_r = _stats.stdev(speaking_rates)
        rate_variation = stdev_r / mean_r if mean_r > 0 else 1.0
        naturalness = max(0.0, 1.0 - rate_variation)
    else:
        naturalness = 1.0

    n_retry = report.get("n_translation_retries", 0)
    retry_rate = max(0.0, 1.0 - n_retry / max(len(metrics), 1))




    # weighted overall score
    weights = {
        "timing_accuracy": 0.30,
        "stretch_quality": 0.25,
        "drift_score": 0.20,
        "naturalness": 0.15,
        "retry_rate": 0.10,
    }


    overall = (
        weights["timing_accuracy"] * timing_accuracy +
        weights["stretch_quality"] * stretch_quality +
        weights["drift_score"] * drift_score +
        weights["naturalness"] * naturalness +
        weights["retry_rate"] * retry_rate
    )

    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "stretch_quality": round(stretch_quality, 3),
        "drift_score": round(drift_score, 3),
        "naturalness": round(naturalness, 3),
        "retry_rate": round(retry_rate, 3),
        "overall": round(overall, 3),
    }