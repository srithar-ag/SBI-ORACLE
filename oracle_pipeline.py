"""
SBI Oracle — Main Pipeline Service
────────────────────────────────────
Orchestrates the full 4-agent sequence for one or all customers:

  1. Data Ingestion       → load customer + CBS data
  2. Macro Signal Agent   → district-level macro score
  3. Behavioral Agent     → 12-signal individual score
  4. Risk Fusion Agent    → Financial Stress Score (0-100)
  5. Intervention Agent   → autonomous action dispatch
  6. Feedback persistence → write results to DB
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.agents.macro_signal_agent import MacroSignalAgent
from app.agents.behavioral_agent import BehavioralPatternAgent
from app.agents.risk_fusion_agent import RiskFusionAgent
from app.agents.intervention_agent import InterventionAgent
from app.models.customer import Customer, RiskBand
from app.models.stress_score import StressScore
from app.models.intervention import Intervention, InterventionType, InterventionChannel, InterventionStatus
from app.config import settings

logger = structlog.get_logger(__name__)


class OraclePipeline:
    """
    Top-level orchestrator. Instantiate once per run; agents are reused across customers.
    """

    def __init__(self) -> None:
        self.macro_agent = MacroSignalAgent()
        self.behavioral_agent = BehavioralPatternAgent()
        self.risk_fusion_agent = RiskFusionAgent()
        self.intervention_agent = InterventionAgent()

    # ── Full-batch run ────────────────────────────────────────────────────────

    async def run_all(self, db: AsyncSession) -> dict:
        """
        Score every active customer. Called by the nightly scheduler.
        Returns a summary dict with counts per risk band.
        """
        logger.info("pipeline.run_all.start")
        result = await db.execute(select(Customer).where(Customer.is_active == True))
        customers = result.scalars().all()

        summary = {"total": len(customers), "red": 0, "amber": 0, "green": 0, "errors": 0}

        # Process in controlled concurrency batches (avoid CBS rate limits)
        batch_size = 50
        for i in range(0, len(customers), batch_size):
            batch = customers[i : i + batch_size]
            tasks = [self._process_customer(c, db) for c in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if isinstance(res, Exception):
                    summary["errors"] += 1
                    logger.error("pipeline.customer_error", error=str(res))
                elif res:
                    summary[res] = summary.get(res, 0) + 1

        logger.info("pipeline.run_all.done", **summary)
        return summary

    # ── Single-customer run ────────────────────────────────────────────────────

    async def run_single(self, customer_id: str, db: AsyncSession) -> dict:
        """
        On-demand scoring for one customer (used by the /score endpoint).
        Returns the fused result as a dict.
        """
        result = await db.execute(select(Customer).where(Customer.id == customer_id))
        customer = result.scalar_one_or_none()
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")

        customer_dict = self._customer_to_dict(customer)
        fused = await self._run_agents(customer, customer_dict)
        await self._persist_score(fused, db)

        # Trigger intervention
        plan = await self.intervention_agent.intervene(fused, customer_dict)
        if plan:
            await self._persist_intervention(plan, fused, db)

        # Update customer record
        customer.current_stress_score = fused.stress_score
        customer.risk_band = fused.risk_band
        customer.last_scored_at = datetime.utcnow()
        await db.commit()

        return {
            "customer_id": customer_id,
            "stress_score": fused.stress_score,
            "risk_band": fused.risk_band,
            "macro_score": fused.macro_score,
            "behavioral_score": fused.behavioral_score,
            "intervention": plan.intervention_type if plan else "monitor_only",
        }

    # ── Internal Helpers ──────────────────────────────────────────────────────

    async def _process_customer(self, customer: Customer, db: AsyncSession) -> str:
        """Score one customer and persist results. Returns risk_band string."""
        customer_dict = self._customer_to_dict(customer)
        fused = await self._run_agents(customer, customer_dict)

        await self._persist_score(fused, db)

        plan = await self.intervention_agent.intervene(fused, customer_dict)
        if plan:
            await self._persist_intervention(plan, fused, db)

        customer.current_stress_score = fused.stress_score
        customer.risk_band = fused.risk_band
        customer.last_scored_at = datetime.utcnow()

        return fused.risk_band

    async def _run_agents(self, customer: Customer, customer_dict: dict):
        """Run Agents 01–03 concurrently (macro + behavioral), then fuse."""
        macro_task = self.macro_agent.assess(customer.district, customer.state)
        behavioral_task = self.behavioral_agent.assess(customer.id, customer_dict)

        macro_result, behavioral_result = await asyncio.gather(macro_task, behavioral_task)

        fused = self.risk_fusion_agent.fuse(customer.id, macro_result, behavioral_result)
        return fused

    async def _persist_score(self, fused, db: AsyncSession) -> StressScore:
        score_record = StressScore(
            customer_id=fused.customer_id,
            score=fused.stress_score,
            risk_band=fused.risk_band,
            macro_score=fused.macro_score,
            behavioral_score=fused.behavioral_score,
            macro_signals=fused.macro_signals,
            behavioral_signals=fused.behavioral_signals,
            model_version=fused.model_version,
        )
        db.add(score_record)
        await db.flush()
        return score_record

    async def _persist_intervention(self, plan, fused, db: AsyncSession) -> None:
        record = Intervention(
            customer_id=plan.customer_id,
            intervention_type=plan.intervention_type,
            channel=plan.channel,
            status=InterventionStatus.SENT,
            payload=plan.offer_details,
            message_text=plan.message_text,
        )
        db.add(record)
        await db.flush()

    @staticmethod
    def _customer_to_dict(c: Customer) -> dict:
        return {
            "id": c.id,
            "name": c.name,
            "phone": c.phone,
            "district": c.district,
            "state": c.state,
            "loan_type": c.loan_type,
            "loan_amount": c.loan_amount,
            "monthly_emi": c.monthly_emi,
            "monthly_income": c.monthly_income,
            "emi_bounce_count_90d": c.emi_bounce_count_90d,
            "salary_delay_days_avg": c.salary_delay_days_avg,
            "unusual_withdrawal_flag": c.unusual_withdrawal_flag,
            "multiple_loan_flag": c.multiple_loan_flag,
        }

    async def close(self) -> None:
        await self.macro_agent.close()
        await self.behavioral_agent.close()
        await self.intervention_agent.close()
