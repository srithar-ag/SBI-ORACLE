"""
SBI Oracle — Agent 02: Behavioral Pattern Agent
────────────────────────────────────────────────
Scans 12 individual-level early-warning signals from SBI CBS:

  1.  EMI bounce count (30/60/90-day windows)
  2.  Salary credit delay (days late vs historical baseline)
  3.  Unusual large cash withdrawals (>3σ from monthly avg)
  4.  Rapid decline in average balance
  5.  Multiple loan applications in 90 days
  6.  Frequent OD (overdraft) utilisation
  7.  ATM withdrawal spike at month-end
  8.  Inward remittance drop (for NRI / migrant worker accounts)
  9.  Utility auto-pay failures
  10. Insurance premium lapse
  11. SIP / RD discontinuation
  12. Spending category shift (essentials↑, discretionary↓)

Returns a BehavioralSignalResult with a behavioral_score (0–100).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# ── Signal definitions ─────────────────────────────────────────────────────────

SIGNAL_WEIGHTS: dict[str, float] = {
    "emi_bounce_90d": 0.18,
    "salary_delay_days": 0.14,
    "unusual_cash_withdrawal": 0.12,
    "avg_balance_decline_pct": 0.10,
    "multiple_loan_applications": 0.10,
    "overdraft_utilisation_pct": 0.08,
    "atm_month_end_spike": 0.06,
    "inward_remittance_drop": 0.06,
    "utility_autopay_failures": 0.08,
    "insurance_lapse_flag": 0.06,
    "sip_rd_discontinuation": 0.06,
    "spending_category_shift": 0.06,
}
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1"


@dataclass
class BehavioralSignalResult:
    customer_id: str
    behavioral_score: float                        # 0–100
    signals: dict = field(default_factory=dict)    # raw values per signal
    triggered_signals: list[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.utcnow)


class BehavioralPatternAgent:
    """
    Agent 02 — scans per-customer account transaction signals
    and returns a weighted behavioral stress score.
    """

    # Normalisation thresholds (above these → score=100 for that signal)
    SIGNAL_MAX: dict[str, float] = {
        "emi_bounce_90d": 3,            # 3+ bounces = max stress
        "salary_delay_days": 20,        # 20+ days late = max stress
        "unusual_cash_withdrawal": 1,   # flag = 0/1
        "avg_balance_decline_pct": 80,  # 80% drop in balance
        "multiple_loan_applications": 3,
        "overdraft_utilisation_pct": 100,
        "atm_month_end_spike": 1,       # flag
        "inward_remittance_drop": 1,    # flag
        "utility_autopay_failures": 3,
        "insurance_lapse_flag": 1,      # flag
        "sip_rd_discontinuation": 1,    # flag
        "spending_category_shift": 1,   # flag
    }

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=10.0)

    # ── Public API ───────────────────────────────────────────────────────────

    async def assess(self, customer_id: str, customer_data: dict | None = None) -> BehavioralSignalResult:
        """
        Main entry point — pulls CBS data and scores 12 behavioral signals.
        `customer_data` can inject pre-fetched data (from DB / seed).
        """
        logger.info("behavioral_agent.assessing", customer_id=customer_id)

        raw_signals = await self._fetch_cbs_signals(customer_id, customer_data)
        score, triggered = self._score_signals(raw_signals)

        result = BehavioralSignalResult(
            customer_id=customer_id,
            behavioral_score=round(score, 2),
            signals=raw_signals,
            triggered_signals=triggered,
        )
        logger.info(
            "behavioral_agent.done",
            customer_id=customer_id,
            behavioral_score=result.behavioral_score,
            triggered=triggered,
        )
        return result

    # ── CBS Data Fetch ────────────────────────────────────────────────────────

    async def _fetch_cbs_signals(
        self, customer_id: str, prefetched: dict | None
    ) -> dict:
        """
        Fetch 12-signal data from SBI CBS REST API.
        Falls back to prefetched DB data or synthetic generation.
        """
        if prefetched:
            return self._extract_from_db_record(prefetched)

        if not settings.sbi_cbs_token:
            return self._synthetic_signals(customer_id)

        try:
            url = f"{settings.sbi_cbs_base_url}/account/{customer_id}/stress-signals"
            resp = await self.client.get(
                url,
                headers={"Authorization": f"Bearer {settings.sbi_cbs_token}"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("behavioral_agent.cbs_fetch_failed", error=str(exc))
            return self._synthetic_signals(customer_id)

    def _extract_from_db_record(self, data: dict) -> dict:
        """Map Customer ORM fields → 12-signal dict."""
        return {
            "emi_bounce_90d": data.get("emi_bounce_count_90d", 0),
            "salary_delay_days": data.get("salary_delay_days_avg", 0),
            "unusual_cash_withdrawal": int(data.get("unusual_withdrawal_flag", False)),
            "avg_balance_decline_pct": data.get("avg_balance_decline_pct", 0),
            "multiple_loan_applications": int(data.get("multiple_loan_flag", False)) * 2,
            "overdraft_utilisation_pct": data.get("overdraft_utilisation_pct", 0),
            "atm_month_end_spike": data.get("atm_month_end_spike", 0),
            "inward_remittance_drop": data.get("inward_remittance_drop", 0),
            "utility_autopay_failures": data.get("utility_autopay_failures", 0),
            "insurance_lapse_flag": data.get("insurance_lapse_flag", 0),
            "sip_rd_discontinuation": data.get("sip_rd_discontinuation", 0),
            "spending_category_shift": data.get("spending_category_shift", 0),
        }

    # ── Score Calculation ─────────────────────────────────────────────────────

    def _score_signals(self, raw: dict) -> tuple[float, list[str]]:
        """
        Normalise each signal to 0–100, apply weights, return composite score.
        """
        normalised: dict[str, float] = {}
        for signal, max_val in self.SIGNAL_MAX.items():
            val = raw.get(signal, 0)
            normalised[signal] = min(100.0, (val / max_val) * 100) if max_val else 0.0

        score = sum(
            normalised[sig] * weight for sig, weight in SIGNAL_WEIGHTS.items()
        )
        score = max(0.0, min(100.0, score))

        triggered = [
            sig for sig, norm_val in normalised.items()
            if norm_val > 50
        ]
        return score, triggered

    # ── Synthetic Fallback ────────────────────────────────────────────────────

    @staticmethod
    def _synthetic_signals(customer_id: str) -> dict:
        seed = sum(ord(c) for c in customer_id)
        rng = random.Random(seed)
        return {
            "emi_bounce_90d": rng.randint(0, 3),
            "salary_delay_days": rng.uniform(0, 25),
            "unusual_cash_withdrawal": rng.randint(0, 1),
            "avg_balance_decline_pct": rng.uniform(0, 70),
            "multiple_loan_applications": rng.randint(0, 3),
            "overdraft_utilisation_pct": rng.uniform(0, 100),
            "atm_month_end_spike": rng.randint(0, 1),
            "inward_remittance_drop": rng.randint(0, 1),
            "utility_autopay_failures": rng.randint(0, 3),
            "insurance_lapse_flag": rng.randint(0, 1),
            "sip_rd_discontinuation": rng.randint(0, 1),
            "spending_category_shift": rng.randint(0, 1),
        }

    async def close(self) -> None:
        await self.client.aclose()
