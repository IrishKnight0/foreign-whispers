"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    # for spanish translations that are too long, this function generates shorter versions to help fit it in the english time window
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    
    logger.info(
        "get_shorter_translations called for %.1fs budget (%d chars baseline) — "
        "returning empty list (student assignment stub).",
        target_duration_s,
        len(baseline_es),
    )
    target_chars = int(target_duration_s * 15)
    candidates = []

    filler_map = [
        ("en este momento", "ahora"),
        ("en realidad", "realmente"),
        ("a pesar de que", "aunque"),
        ("debido a que", "porque"),
        ("con el fin de", "para"),
        ("a causa de", "por"),
        ("sin embargo", "pero"),
        ("por lo tanto", "entonces"),
        ("es decir", "o sea"),
        ("a continuación", "luego"),
        ("en primer lugar", "primero"),
        ("en segundo lugar", "segundo"),
        ("por otro lado", "además"),
        ("de todas formas", "igual"),
        ("en definitiva", "en fin"),
    ]

    result_t = baseline_es
    used = []
    for long_form, short_form in filler_map:
        if long_form in result_t.lower():
            result_t = result_t.lower().replace(long_form, short_form)
            used.append(f"{long_form}→{short_form}")

    if result_t != baseline_es and len(result_t) <= len(baseline_es):
        candidates.append(TranslationCandidate(
            text=result_t,
            char_count=len(result_t),
            brevity_rationale=f"Contracted phrases: {', '.join(used)}",
        ))

    if len(baseline_es) > target_chars:
        words = baseline_es.split()
        shortVer = ""
        for word in words:
            if len(shortVer) + len(word) + 1 <= target_chars:
                shortVer += (" " if shortVer else "") + word
            else:
                break
        if shortVer and shortVer != baseline_es:
            candidates.append(TranslationCandidate(
                text=shortVer,
                char_count=len(shortVer),
                brevity_rationale="Truncated to word boundary within duration budget",
            ))

    import re

    cleaned = re.sub(r'\([^)]*\)', '', baseline_es).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if cleaned != baseline_es and len(cleaned) > 0:
        candidates.append(TranslationCandidate(
            text=cleaned,
            char_count=len(cleaned),
            brevity_rationale="Removed parenthetical clauses",
        ))

    candidates.sort(key=lambda c: c.char_count)
    return candidates
