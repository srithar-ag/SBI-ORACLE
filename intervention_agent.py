"""
SBI Oracle — Agent 04: Intervention Agent
──────────────────────────────────────────
Autonomously triggers the right intervention based on risk band:

  🔴 Red  ≥ 70  → Restructuring offer / Emergency credit / Branch manager escalation
  🟡 Amber 40–69 → Soft nudge via YONO / WhatsApp
  🟢 Green < 40  → Monitor only (no outreach)

Uses Claude to generate personalised, empathetic messages in the
customer's preferred language (defaults to English / Hindi).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import structlog

from app.config import settings
from app.agents.risk_fusion_agent import FusedRiskResult

logger = structlog.get_logger(__name__)


@dataclass
class InterventionPlan:
    customer_id: str
    risk_band: str
    intervention_type: str
    channel: str
    message_text: str
    offer_details: dict = field(default_factory=dict)
    triggered_at: datetime = field(default_factory=datetime.utcnow)


class InterventionAgent:
    """
    Agent 04 — decides intervention type, generates a personalised message
    via the Claude API, and dispatches it over the appropriate channel.
    """

    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=30.0)

    # ── Public API ───────────────────────────────────────────────────────────

    async def intervene(
        self,
        fused: FusedRiskResult,
        customer: dict,
    ) -> InterventionPlan | None:
        """
        Main entry point.
        Returns None if risk_band is green (monitor-only).
        """
        logger.info(
            "intervention_agent.start",
            customer_id=fused.customer_id,
            band=fused.risk_band,
        )

        if fused.risk_band == "green":
            logger.info("intervention_agent.monitor_only", customer_id=fused.customer_id)
            return None

        plan = await self._build_plan(fused, customer)
        await self._dispatch(plan)

        logger.info(
            "intervention_agent.dispatched",
            customer_id=fused.customer_id,
            type=plan.intervention_type,
            channel=plan.channel,
        )
        return plan

    # ── Plan Builder ─────────────────────────────────────────────────────────

    async def _build_plan(
        self, fused: FusedRiskResult, customer: dict
    ) -> InterventionPlan:
        intervention_type, channel, offer = self._decide(fused, customer)
        message = await self._generate_message(fused, customer, intervention_type, offer)

        return InterventionPlan(
            customer_id=fused.customer_id,
            risk_band=fused.risk_band,
            intervention_type=intervention_type,
            channel=channel,
            message_text=message,
            offer_details=offer,
        )

    def _decide(
        self, fused: FusedRiskResult, customer: dict
    ) -> tuple[str, str, dict]:
        """
        Rule-based decision tree → (intervention_type, channel, offer_details).
        """
        loan_amount = customer.get("loan_amount", 0)
        monthly_emi = customer.get("monthly_emi", 0)

        if fused.risk_band == "red":
            if loan_amount > 25_00_000:  # > 25L → high value, human escalation
                return (
                    "restructuring_offer",
                    "branch_manager",
                    {
                        "emi_reduction_pct": 30,
                        "tenure_extension_months": 12,
                        "moratorium_months": 3,
                    },
                )
            elif "agriculture" in customer.get("loan_type", ""):
                return (
                    "micro_insurance",
                    "yono",
                    {"insurance_cover": min(monthly_emi * 6, 1_00_000)},
                )
            else:
                return (
                    "restructuring_offer",
                    "whatsapp",
                    {
                        "emi_reduction_pct": 20,
                        "tenure_extension_months": 6,
                    },
                )

        # Amber band
        return (
            "soft_nudge",
            "yono",
            {"financial_health_tip": True},
        )

    # ── LLM Message Generation ────────────────────────────────────────────────

    async def _generate_message(
        self,
        fused: FusedRiskResult,
        customer: dict,
        intervention_type: str,
        offer: dict,
    ) -> str:
        """
        Call Claude to generate an empathetic, personalised message
        in plain language — no bank jargon.
        """
        if not settings.anthropic_api_key:
            return self._fallback_message(intervention_type, customer, offer)

        prompt = self._build_llm_prompt(fused, customer, intervention_type, offer)

        try:
            resp = await self.http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"].strip()
        except Exception as exc:
            logger.warning("intervention_agent.llm_failed", error=str(exc))
            return self._fallback_message(intervention_type, customer, offer)

    @staticmethod
    def _build_llm_prompt(
        fused: FusedRiskResult,
        customer: dict,
        intervention_type: str,
        offer: dict,
    ) -> str:
        return f"""You are SBI Oracle, an empathetic financial wellness assistant at State Bank of India.

Write a SHORT (≤3 sentences), warm, non-alarming message to a customer who may be experiencing 
financial stress. DO NOT mention "NPA", "default", or "risk score". Be supportive and solution-focused.

Customer context:
- Name: {customer.get('name', 'Valued Customer')}
- Loan type: {customer.get('loan_type', 'loan')}
- EMI: ₹{customer.get('monthly_emi', 0):,.0f}/month
- Intervention: {intervention_type}
- Offer: {json.dumps(offer)}

Write ONLY the message text. No subject line, no salutation like "Dear".
Start with the customer's first name."""

    @staticmethod
    def _fallback_message(
        intervention_type: str,
        customer: dict,
        offer: dict,
    ) -> str:
        name = customer.get("name", "Valued Customer").split()[0]
        if intervention_type == "restructuring_offer":
            return (
                f"{name}, we noticed your account may need some support. "
                f"SBI is offering a special EMI restructuring plan — "
                f"lower instalments for {offer.get('tenure_extension_months', 6)} months. "
                "Reply YES to know more."
            )
        if intervention_type == "soft_nudge":
            return (
                f"{name}, a quick check — your SBI account shows some activity patterns. "
                "We're here if you need financial guidance. Visit your nearest branch or call 1800-11-2211."
            )
        return f"{name}, SBI Oracle is here to help. Contact us at 1800-11-2211."

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, plan: InterventionPlan) -> None:
        """Send message over the chosen channel."""
        channel_map = {
            "yono": self._send_yono,
            "whatsapp": self._send_whatsapp,
            "sms": self._send_sms,
            "branch_manager": self._escalate_to_branch,
        }
        handler = channel_map.get(plan.channel, self._send_sms)
        await handler(plan)

    async def _send_yono(self, plan: InterventionPlan) -> None:
        if not settings.sbi_yono_api_url:
            logger.info("intervention_agent.yono_mock", customer_id=plan.customer_id)
            return
        try:
            await self.http.post(
                f"{settings.sbi_yono_api_url}/push",
                json={"customer_id": plan.customer_id, "message": plan.message_text},
                headers={"Authorization": f"Bearer {settings.sbi_cbs_token}"},
            )
        except Exception as exc:
            logger.error("intervention_agent.yono_failed", error=str(exc))

    async def _send_whatsapp(self, plan: InterventionPlan) -> None:
        logger.info(
            "intervention_agent.whatsapp_mock",
            customer_id=plan.customer_id,
            msg_preview=plan.message_text[:60],
        )

    async def _send_sms(self, plan: InterventionPlan) -> None:
        logger.info(
            "intervention_agent.sms_mock",
            customer_id=plan.customer_id,
            msg_preview=plan.message_text[:60],
        )

    async def _escalate_to_branch(self, plan: InterventionPlan) -> None:
        logger.info(
            "intervention_agent.branch_escalation",
            customer_id=plan.customer_id,
            offer=plan.offer_details,
        )

    async def close(self) -> None:
        await self.http.aclose()
