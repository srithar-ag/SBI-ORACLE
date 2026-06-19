# SBI Oracle 🔮
### Proactive Financial Distress Prediction Engine
> *Team Outliers | SBI Hackathon @ HackCulture 2026 | Agentic AI & Emerging Tech*

---

## Problem
India's ₹5L Cr+ NPA crisis is **not a default problem — it's a prediction problem.**  
Every NPA starts with 60–90 days of warning signals that go undetected.  
SBI Oracle doesn't ignore them.

---

## Solution
An **autonomous multi-agent AI system** that:
1. Detects financial distress **60–90 days early**
2. Scores each borrower with a **Financial Stress Score (0–100)**
3. Automatically sends **personalised intervention** via YONO / WhatsApp / SMS
4. Continuously improves via a **feedback loop**

---

## Architecture — 4 Agents Working in Sequence

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SBI Oracle Pipeline                          │
│                                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ 01 Macro     │   │ 02 Behavioral│   │ 03 Risk      │            │
│  │ Signal Agent │──▶│ Pattern Agent│──▶│ Fusion Agent │──▶ Score   │
│  │              │   │              │   │              │            │
│  │ RBI DBIE API │   │ 12 Signals   │   │ XGBoost +    │            │
│  │ IMD API      │   │ from CBS     │   │ LSTM         │            │
│  │ Employment   │   │              │   │ 0–100 score  │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│                                                  │                   │
│                                                  ▼                   │
│                                     ┌──────────────────┐            │
│                                     │ 04 Intervention  │            │
│                                     │ Agent            │            │
│                                     │ 🔴 Red → Restr.  │            │
│                                     │ 🟡 Amber → Nudge │            │
│                                     │ 🟢 Green → Watch │            │
│                                     └──────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
sbi-oracle/
├── backend/                    # Python / FastAPI
│   ├── app/
│   │   ├── agents/             # 4 specialised AI agents
│   │   │   ├── macro_signal_agent.py
│   │   │   ├── behavioral_agent.py
│   │   │   ├── risk_fusion_agent.py
│   │   │   └── intervention_agent.py
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── routers/            # FastAPI route handlers
│   │   ├── services/           # Pipeline orchestration + data gen
│   │   ├── config.py           # Centralised settings (pydantic)
│   │   ├── database.py         # Async DB engine + session
│   │   └── main.py             # FastAPI app entry point
│   ├── ml/models/              # Trained XGBoost / LSTM model files
│   ├── data/synthetic/         # Generated demo data
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                   # React / TypeScript / Tailwind
│   ├── src/
│   │   ├── components/         # Dashboard, CustomerTable, Heatmap, etc.
│   │   ├── pages/
│   │   ├── services/api.ts     # Axios API client
│   │   └── types/index.ts      # Shared TypeScript types
│   ├── package.json
│   └── .env.example
│
├── scripts/                    # CLI utilities
├── docs/
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Multi-Agent Framework | LangGraph + CrewAI |
| LLM | Claude API (claude-sonnet-4-6) |
| Macro Data | RBI DBIE API + IMD Public API |
| Banking Data | SBI CBS via internal REST |
| ML Models | XGBoost + LSTM |
| Backend | Python + FastAPI + PostgreSQL + Redis |
| Communication | YONO API + WhatsApp API + MSG91 |
| Deployment | AWS GovCloud / Azure (MEITY compliant) |
| Frontend | React + TypeScript + Tailwind + Recharts |

---

## Quick Start

### Backend

```bash
cd backend

# 1. Copy env file and fill in your keys
cp .env.example .env

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --reload --port 8000

# 5. Seed 500 synthetic customers
curl -X POST "http://localhost:8000/api/v1/admin/seed?n=500"
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173

---

## Risk Band Thresholds

| Band | Score | Action |
|---|---|---|
| 🔴 Red | ≥ 70 | Immediate: restructuring offer / emergency credit / branch escalation |
| 🟡 Amber | 40–69 | Gentle nudge via YONO / WhatsApp |
| 🟢 Green | < 40 | Monitor only — no outreach |

---

## Business Impact

- **25–30%** reduction in early-stage NPA formation
- **₹300–500 Cr** annual provisioning cost savings at SBI scale
- **₹25–40K** Customer Lifetime Value retained per avoided default
- **₹2,000 Cr+** addressable SaaS licensing market across Indian PSU banks

---

## Team Outliers

| Member | Responsibilities |
|---|---|
| Padma Sritharan AG | Multi-Agent Framework, Macro Data Sources, Banking Data Layer |
| Praveen R | LLM Layer, Communication Layer, Databases |
| Aadhithya P | ML Models, Backend, Deployment |
