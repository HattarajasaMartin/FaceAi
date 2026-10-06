from __future__ import annotations

from .config import display_name, score_percent
from .domain import RecognitionResult, VerifyResult


def decide_faceid(result: RecognitionResult) -> VerifyResult:
    if result.status == "cocok" and result.best_name:
        label = display_name(result.best_name)
        return VerifyResult(
            accepted=True,
            code="TERIMA_FACEID",
            claimed="faceid",
            best_name=result.best_name,
            claimed_score=result.best_score,
            best_score=result.best_score,
            scores=result.scores,
            message=f"TERIMA {label}  {score_percent(result.best_score)}%",
        )
    message = result.message
    if result.status != "tidak_dikenali" or "Belum ada" not in message:
        message = "TOLAK: tidak dikenali"
    return VerifyResult(
        accepted=False,
        code="TOLAK_FACEID",
        claimed="faceid",
        best_name=result.best_name,
        claimed_score=result.best_score,
        best_score=result.best_score,
        scores=result.scores,
        message=message,
    )
