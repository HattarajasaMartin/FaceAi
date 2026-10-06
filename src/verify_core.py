from __future__ import annotations

from .config import IMPOSTOR_MARGIN, MATCH_THRESHOLD, display_name, score_percent
from .domain import NameScore, VerifyResult


def decide_verify(
    claimed: str,
    scores: tuple[NameScore, ...],
    enrolled: frozenset[str],
    *,
    threshold: float = MATCH_THRESHOLD,
    margin: float = IMPOSTOR_MARGIN,
) -> VerifyResult:
    claimed_label = display_name(claimed)
    score_map = {item.name: item.score for item in scores}
    claimed_score = score_map.get(claimed, 0.0)
    best = scores[0] if scores else None
    best_name = best.name if best else None
    best_score = best.score if best else 0.0

    if claimed not in enrolled:
        return VerifyResult(
            accepted=False,
            code="TOLAK_UNENROLLED",
            claimed=claimed,
            best_name=best_name,
            claimed_score=claimed_score,
            best_score=best_score,
            scores=scores,
            message=f"{claimed_label} belum terdaftar",
        )

    others = [item for item in scores if item.name != claimed and item.name in enrolled]
    rival = others[0] if others else None

    if rival and rival.score >= threshold and rival.score > claimed_score + margin:
        return VerifyResult(
            accepted=False,
            code="TOLAK_IMPERSONATION",
            claimed=claimed,
            best_name=rival.name,
            claimed_score=claimed_score,
            best_score=rival.score,
            scores=scores,
            message=(
                f"TOLAK: bukan {claimed_label}. Wajah lebih mirip {display_name(rival.name)}"
            ),
        )

    if claimed_score < threshold:
        return VerifyResult(
            accepted=False,
            code="TOLAK_MISMATCH",
            claimed=claimed,
            best_name=best_name,
            claimed_score=claimed_score,
            best_score=best_score,
            scores=scores,
            message=f"TOLAK: wajah tidak cocok dengan {claimed_label}",
        )

    return VerifyResult(
        accepted=True,
        code="TERIMA",
        claimed=claimed,
        best_name=claimed,
        claimed_score=claimed_score,
        best_score=claimed_score,
        scores=scores,
        message=f"TERIMA {claimed_label}  {score_percent(claimed_score)}%",
    )
