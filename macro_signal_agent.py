"""
SBI Oracle — Agent 01: Macro Signal Agent
─────────────────────────────────────────
Monitors district-level macroeconomic indicators from:
  • RBI DBIE API   — credit growth, repo rate, sectoral stress
  • IMD API        — rainfall deficit, flood/drought advisories
  • Commodity APIs — crop & commodity price indices
  • Govt datasets  — CMIE district employment trends

Returns a MacroSignalResult with a geo-weighted macro_score (0–100).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class MacroSignalResult:
    district: str
    state: str
    macro_score: float                          # 0–100 (higher = more stressed)
    signals: dict = field(default_factory=dict) # raw signal breakdown
    risk_factors: list[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.utcnow)


class MacroSignalAgent:
    """
    Agent 01 — pulls macro / geographic risk data and computes
    a district-level economic stress index.
    """

    RBI_SECTORS_AT_RISK = {"agriculture", "msme", "construction", "textiles"}

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=10.0)

    # ── Public API ───────────────────────────────────────────────────────────

    async def assess(self, district: str, state: str) -> MacroSignalResult:
        """
        Main entry point — returns MacroSignalResult for a given district.
        Falls back to synthetic signals when live APIs are unavailable.
        """
        logger.info("macro_agent.assessing", district=district, state=state)

        rbi_data = await self._fetch_rbi_signals(district, state)
        imd_data = await self._fetch_imd_signals(district)
        employment_data = await self._fetch_employment_signals(district, state)
        commodity_data = await self._fetch_commodity_signals(state)

        score, signals, risk_factors = self._fuse_signals(
            rbi_data, imd_data, employment_data, commodity_data
        )

        result = MacroSignalResult(
            district=district,
            state=state,
            macro_score=round(score, 2),
            signals=signals,
            risk_factors=risk_factors,
        )
        logger.info(
            "macro_agent.done",
            district=district,
            macro_score=result.macro_score,
            risk_factors=risk_factors,
        )
        return result

    # ── Private Fetchers (with synthetic fallback) ────────────────────────────

    async def _fetch_rbi_signals(self, district: str, state: str) -> dict:
        """
        Pull credit-related stress indicators from RBI DBIE.
        Falls back to synthetic data if API is unreachable.
        """
        if not settings.rbi_dbie_api_key:
            return self._synthetic_rbi(district)

        try:
            url = f"{settings.rbi_dbie_base_url}/api/credit/district"
            resp = await self.client.get(
                url,
                params={"district": district, "state": state},
                headers={"X-API-KEY": settings.rbi_dbie_api_key},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("macro_agent.rbi_fetch_failed", error=str(exc))
            return self._synthetic_rbi(district)

    async def _fetch_imd_signals(self, district: str) -> dict:
        """Pull weather / climate stress data from IMD."""
        if not settings.imd_api_key:
            return self._synthetic_imd(district)

        try:
            url = f"{settings.imd_api_base_url}/district-rainfall"
            resp = await self.client.get(
                url,
                params={"district": district},
                headers={"Authorization": f"Bearer {settings.imd_api_key}"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("macro_agent.imd_fetch_failed", error=str(exc))
            return self._synthetic_imd(district)

    async def _fetch_employment_signals(self, district: str, state: str) -> dict:
        """CMIE employment district data (proxied via internal data lake)."""
        # In production: query internal data lake / warehouse
        return self._synthetic_employment(district)

    async def _fetch_commodity_signals(self, state: str) -> dict:
        """Agri commodity price stress (NCDEX / eNAM indices)."""
        return self._synthetic_commodity(state)

    # ── Score Fusion ─────────────────────────────────────────────────────────

    def _fuse_signals(
        self,
        rbi: dict,
        imd: dict,
        employment: dict,
        commodity: dict,
    ) -> tuple[float, dict, list[str]]:
        """
        Weighted combination of macro sub-scores → single macro_score 0–100.
        Weights reflect RBI's Priority Sector stress research.
        """
        weights = {
            "credit_stress": 0.35,
            "rainfall_deficit": 0.20,
            "employment_stress": 0.25,
            "commodity_price_fall": 0.20,
        }

        sub_scores = {
            "credit_stress": rbi.get("credit_stress_index", 0),
            "rainfall_deficit": imd.get("rainfall_deficit_pct", 0),
            "employment_stress": employment.get("unemployment_rate_index", 0),
            "commodity_price_fall": commodity.get("commodity_stress_index", 0),
        }

        score = sum(sub_scores[k] * w for k, w in weights.items())
        score = max(0.0, min(100.0, score))

        risk_factors = [
            k.replace("_", " ").title()
            for k, v in sub_scores.items()
            if v > 60
        ]

        signals = {**sub_scores, "rbi_raw": rbi, "imd_raw": imd}
        return score, signals, risk_factors

    # ── Synthetic Fallbacks (dev / demo mode) ─────────────────────────────────

    @staticmethod
    def _synthetic_rbi(district: str) -> dict:
        seed = sum(ord(c) for c in district)
        rng = random.Random(seed)
        return {
            "credit_stress_index": rng.uniform(10, 80),
            "npa_growth_rate_pct": rng.uniform(0, 15),
            "priority_sector_stress": rng.choice([True, False]),
        }

    @staticmethod
    def _synthetic_imd(district: str) -> dict:
        seed = sum(ord(c) for c in district) + 7
        rng = random.Random(seed)
        return {
            "rainfall_deficit_pct": rng.uniform(0, 90),
            "drought_advisory": rng.random() > 0.7,
            "flood_alert": rng.random() > 0.85,
        }

    @staticmethod
    def _synthetic_employment(district: str) -> dict:
        seed = sum(ord(c) for c in district) + 13
        rng = random.Random(seed)
        return {"unemployment_rate_index": rng.uniform(5, 75)}

    @staticmethod
    def _synthetic_commodity(state: str) -> dict:
        seed = sum(ord(c) for c in state) + 19
        rng = random.Random(seed)
        return {"commodity_stress_index": rng.uniform(0, 70)}

    async def close(self) -> None:
        await self.client.aclose()
