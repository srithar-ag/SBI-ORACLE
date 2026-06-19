"""
SBI Oracle — Agent 03: Risk Fusion Agent
─────────────────────────────────────────
Combines outputs from the Macro Signal Agent and Behavioral Pattern Agent
into a single Financial Stress Score (0–100) per customer using an
XGBoost model (with LSTM temporal smoothing in production).

Score bands:
  🔴 Red    ≥ 70  → immediate intervention
  🟡 Amber  40–69 → gentle nudge
  🟢 Green  < 40  → monitor only
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import structlog

from app.config import settings
from app.agents.macro_signal_agent import MacroSignalResult
from app.agents.behavioral_agent import BehavioralSignalResult

logger = structlog.get_logger(__name__)


@dataclass
class FusedRiskResult:
    customer_id: str
    stress_score: float             # 0–100 composite
    risk_band: str                  # green / amber / red
    macro_score: float
    behavioral_score: float
    macro_signals: dict = field(default_factory=dict)
    behavioral_signals: dict = field(default_factory=dict)
    model_version: str = "v1.0.0"
    fused_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def requires_immediate_action(self) -> bool:
        return self.risk_band == "red"

    @property
    def requires_nudge(self) -> bool:
        return self.risk_band == "amber"


class RiskFusionAgent:
    """
    Agent 03 — fuses macro + behavioral scores using an ML model
    (XGBoost in production; weighted average fallback in dev).
    """

    # Feature order expected by the XGBoost model
    FEATURE_NAMES = [
        "macro_score",
        "behavioral_score",
        "emi_bounce_90d",
        "salary_delay_days",
        "unusual_cash_withdrawal",
        "avg_balance_decline_pct",
        "credit_stress_index",
        "rainfall_deficit_pct",
        "unemployment_rate_index",
        "commodity_stress_index",
        "multiple_loan_applications",
        "overdraft_utilisation_pct",
    ]

    def __init__(self) -> None:
        self._model = self._load_model()

    # ── Public API ───────────────────────────────────────────────────────────

    def fuse(
        self,
        customer_id: str,
        macro: MacroSignalResult,
        behavioral: BehavioralSignalResult,
    ) -> FusedRiskResult:
        """
        Produce a fused Financial Stress Score from both agent outputs.
        """
        logger.info("risk_fusion.fusing", customer_id=customer_id)

        score = self._predict(macro, behavioral)
        band = self._band(score)

        result = FusedRiskResult(
            customer_id=customer_id,
            stress_score=round(score, 2),
            risk_band=band,
            macro_score=macro.macro_score,
            behavioral_score=behavioral.behavioral_score,
            macro_signals=macro.signals,
            behavioral_signals=behavioral.signals,
            model_version=settings.model_version,
        )
        logger.info(
            "risk_fusion.done",
            customer_id=customer_id,
            score=result.stress_score,
            band=band,
        )
        return result

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        model_path = Path(settings.xgboost_model_path)
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                logger.info("risk_fusion.model_loaded", path=str(model_path))
                return model
            except Exception as exc:
                logger.warning("risk_fusion.model_load_failed", error=str(exc))
        logger.warning("risk_fusion.using_fallback_scorer")
        return None

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _predict(
        self,
        macro: MacroSignalResult,
        behavioral: BehavioralSignalResult,
    ) -> float:
        if self._model is not None:
            return self._xgboost_predict(macro, behavioral)
        return self._weighted_fallback(macro, behavioral)

    def _xgboost_predict(
        self,
        macro: MacroSignalResult,
        behavioral: BehavioralSignalResult,
    ) -> float:
        """Run inference via the trained XGBoost model."""
        beh = behavioral.signals
        mac = macro.signals

        features = np.array([[
            macro.macro_score,
            behavioral.behavioral_score,
            beh.get("emi_bounce_90d", 0),
            beh.get("salary_delay_days", 0),
            beh.get("unusual_cash_withdrawal", 0),
            beh.get("avg_balance_decline_pct", 0),
            mac.get("credit_stress_index", 0),
            mac.get("rainfall_deficit_pct", 0),
            mac.get("unemployment_rate_index", 0),
            mac.get("commodity_stress_index", 0),
            beh.get("multiple_loan_applications", 0),
            beh.get("overdraft_utilisation_pct", 0),
        ]])

        raw = float(self._model.predict(features)[0])
        # Model output is already 0-100; clamp defensively
        return max(0.0, min(100.0, raw))

    @staticmethod
    def _weighted_fallback(
        macro: MacroSignalResult,
        behavioral: BehavioralSignalResult,
    ) -> float:
        """
        Simple weighted average fallback — used in dev/demo mode
        when the XGBoost model file is absent.
        Behavioral signals are weighted higher (they're more predictive
        of individual default risk than macro alone).
        """
        return macro.macro_score * 0.35 + behavioral.behavioral_score * 0.65

    # ── Band Classification ────────────────────────────────────────────────────

    @staticmethod
    def _band(score: float) -> str:
        if score >= settings.stress_score_red_threshold:
            return "red"
        if score >= settings.stress_score_amber_threshold:
            return "amber"
        return "green"
