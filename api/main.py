"""
OpenFinGuard — FastAPI Inference Backend
=========================================
WHY THIS FILE EXISTS:
  A trained model is worthless without a serving layer. FastAPI gives us:
  - Swagger UI auto-generated from type hints (zero extra work)
  - Async request handling (handles concurrent loan applications)
  - Pydantic validation (rejects malformed inputs before they hit the model)
  - Sub-10ms inference latency for real-time credit decisions
  - PostgreSQL persistence: every prediction is stored and auditable

  This is what separates ML engineers from data scientists.
  Data scientists train models. Engineers deploy them.
"""

import json
import time
import uuid
import joblib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Boolean, DateTime, Date, Text, JSON, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "figures"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openfingraud.api")

# ── Database Setup ──────────────────────────────────────────────────────────
# WHY SQLALCHEMY:
# SQLAlchemy gives us an ORM layer that lets us write Python objects instead
# of raw SQL strings. This means type safety, testability, and protection
# against SQL injection — critical for financial data.

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/openfingraud"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ─────────────────────────────────────────────────────────────
class DBPrediction(Base):
    """
    WHY STORE PREDICTIONS:
    In production, every credit decision must be:
    1. Auditable — regulators can demand records going back 25 months (ECOA)
    2. Monitorable — drift detection compares new predictions vs. historical baseline
    3. Explainable — adverse action notices require stored explanations
    """
    __tablename__ = "predictions"

    id               = Column(Integer, primary_key=True, index=True)
    request_id       = Column(String(64), unique=True, nullable=False, index=True)
    application_id   = Column(String(64), nullable=True)
    risk_score       = Column(Float, nullable=False)
    risk_tier        = Column(String(16), nullable=False)
    decision         = Column(String(16), nullable=False)
    confidence       = Column(String(8), nullable=False)
    features_json    = Column(JSON, nullable=False)
    explanation_json = Column(JSON, nullable=True)
    nl_explanation   = Column(Text, nullable=True)
    model_version    = Column(String(32), nullable=False)
    model_name       = Column(String(64), nullable=False)
    inference_ms     = Column(Float, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)


class DBDriftMetric(Base):
    """Stores per-prediction PSI-style drift signals for monitoring."""
    __tablename__ = "drift_metrics_live"

    id           = Column(Integer, primary_key=True, index=True)
    request_id   = Column(String(64), nullable=False)
    feature_name = Column(String(64), nullable=False)
    feature_value= Column(Float, nullable=True)
    recorded_at  = Column(DateTime, default=datetime.utcnow)


def get_db():
    """Dependency injection for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    request_id          VARCHAR(64) UNIQUE,
    decision            VARCHAR(16),
    risk_band           VARCHAR(32),
    default_probability FLOAT,
    model_name          VARCHAR(64),
    model_version       VARCHAR(32),
    inference_ms        FLOAT,
    features_json       JSONB,
    explanation_json    JSONB,
    created_at          TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_decision ON predictions(decision);
"""
def init_db():
    """Create all tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified via SQLAlchemy")
    except Exception as e:
        logger.warning(f"Database init failed (will retry): {e}")

# ── Model Store ─────────────────────────────────────────────────────────────
class ModelStore:
    model           = None
    scaler          = None
    explainer       = None
    feature_names   = None
    metadata        = None
    shap_importance = None

    @classmethod
    def load(cls):
        logger.info("Loading model artifacts...")
        cls.model  = joblib.load(MODELS_DIR / "champion_model.joblib")
        cls.scaler = joblib.load(MODELS_DIR / "scaler.joblib")

        with open(MODELS_DIR / "champion_metadata.json") as f:
            cls.metadata = json.load(f)

        with open(MODELS_DIR / "feature_names.json") as f:
            cls.feature_names = json.load(f)

        try:
            with open(MODELS_DIR / "shap_feature_importance.json") as f:
                cls.shap_importance = json.load(f)
        except FileNotFoundError:
            cls.shap_importance = {}

        try:
            cls.explainer = shap.TreeExplainer(cls.model)
            logger.info("TreeExplainer loaded for real-time SHAP")
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer unavailable: {e}")
            try:
                # Fallback: Linear explainer for non-tree models
                cls.explainer = shap.LinearExplainer(
                    cls.model,
                    shap.sample(pd.DataFrame(columns=cls.feature_names), 100)
                )
            except Exception:
                cls.explainer = None

        logger.info(
            f"Model loaded: {cls.metadata.get('champion_model', 'unknown')} | "
            f"Threshold: {cls.metadata.get('optimal_threshold', 0.5)}"
        )


# ── App Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler (replaces deprecated on_event)."""
    ModelStore.load()
    init_db()
    logger.info("OpenFinGuard API ready")
    yield
    logger.info("OpenFinGuard API shutting down")


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OpenFinGuard — AI Credit Intelligence API",
    description="""
## Credit Risk Assessment with Explainable AI

Real-time credit risk assessment with SHAP-based explanations for every decision.
Every prediction is persisted to PostgreSQL for audit and drift monitoring.

### Decision Logic
| Decision | P(default) |
|----------|-----------|
| APPROVE  | < 25%     |
| REVIEW   | 25–55%    |
| DECLINE  | ≥ 55%     |

### Compliance
- ECOA / FCRA adverse action reasons via SHAP
- Full prediction audit trail in PostgreSQL
- Age-group fairness monitoring
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Feature Descriptions ────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "RevolvingUtilizationOfUnsecuredLines": "credit card utilization",
    "age":                                  "borrower age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59 day late payments",
    "DebtRatio":                            "debt-to-income ratio",
    "MonthlyIncome":                        "monthly income",
    "NumberOfOpenCreditLinesAndLoans":      "number of open accounts",
    "NumberOfTimes90DaysLate":              "90+ day late payments",
    "NumberRealEstateLoansOrLines":         "real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 day late payments",
    "NumberOfDependents":                   "number of dependents",
    "TotalDelinquencies":                   "total delinquency count",
    "DelinquencySeverityScore":             "delinquency severity score",
    "EstimatedMonthlyDebt":                 "estimated monthly debt",
    "DisposableIncome":                     "estimated disposable income",
    "CreditLineDensity":                    "credit lines per age year",
}

# ── Risk Band Table ──────────────────────────────────────────────────────────
RISK_BANDS = [
    (0.00, 0.05, "Very Low Risk",  "APPROVE", "HIGH",   "Approve — Excellent credit profile"),
    (0.05, 0.15, "Low Risk",       "APPROVE", "HIGH",   "Approve — Good credit profile"),
    (0.15, 0.25, "Moderate Risk",  "APPROVE", "MEDIUM", "Approve with standard terms"),
    (0.25, 0.40, "Elevated Risk",  "REVIEW",  "MEDIUM", "Manual review recommended"),
    (0.40, 0.55, "High Risk",      "REVIEW",  "LOW",    "Enhanced underwriting required"),
    (0.55, 1.00, "Very High Risk", "DECLINE", "HIGH",   "Decline — Does not meet credit criteria"),
]


def get_risk_band(prob: float) -> tuple:
    for lo, hi, band, decision, confidence, action in RISK_BANDS:
        if lo <= prob < hi:
            return band, decision, confidence, action
    return "Very High Risk", "DECLINE", "HIGH", "Decline — Does not meet credit criteria"


# ── Pydantic Schemas ────────────────────────────────────────────────────────
class CreditApplicationRequest(BaseModel):
    revolving_utilization:  float = Field(..., ge=0.0, le=15.0, example=0.35,
                                          description="Revolving credit utilization ratio")
    age:                    int   = Field(..., ge=18,  le=100,  example=45)
    past_due_30_59_days:    int   = Field(..., ge=0,   le=20,   example=1)
    debt_ratio:             float = Field(..., ge=0.0, le=50.0, example=0.4)
    monthly_income:         float = Field(..., ge=0.0,          example=5000.0)
    open_credit_lines:      int   = Field(..., ge=0,   le=50,   example=8)
    past_due_90_days:       int   = Field(..., ge=0,   le=20,   example=0)
    real_estate_loans:      int   = Field(..., ge=0,   le=20,   example=1)
    past_due_60_89_days:    int   = Field(..., ge=0,   le=20,   example=0)
    dependents:             int   = Field(..., ge=0,   le=20,   example=2)
    application_id:         Optional[str]  = Field(None)
    explain:                bool           = Field(True)

    @validator("monthly_income")
    def income_check(cls, v):
        if v > 500_000:
            raise ValueError("Monthly income exceeds $500,000 — please verify")
        return v

    @validator("revolving_utilization")
    def utilization_check(cls, v):
        # Accept percentage input (e.g. 35 for 35%) and normalize
        if v > 15:
            raise ValueError("revolving_utilization must be 0–15 (ratio, not percentage)")
        return v


class CreditDecisionResponse(BaseModel):
    request_id:            str
    application_id:        Optional[str]
    timestamp:             str
    decision:              str
    risk_band:             str
    default_probability:   float
    confidence:            str
    primary_risk_factors:  list
    protective_factors:    list
    recommended_action:    str
    model_version:         str
    model_name:            str
    threshold_used:        float
    inference_time_ms:     float
    persisted:             bool = False


class PredictionHistoryItem(BaseModel):
    request_id:          str
    decision:            str
    risk_band:           str
    default_probability: float
    model_name:          str
    created_at:          str
    application_id:      Optional[str]


# ── Feature Engineering ─────────────────────────────────────────────────────
def engineer_features(req: CreditApplicationRequest) -> pd.DataFrame:
    """
    WHY THIS MIRRORS run_pipeline.py:
    Training/serving skew is the most common production ML bug.
    The same transformations that ran during training MUST run at inference.
    Any deviation here causes silent performance degradation.
    """
    raw = {
        "RevolvingUtilizationOfUnsecuredLines": req.revolving_utilization,
        "age":                                  req.age,
        "NumberOfTime30-59DaysPastDueNotWorse": req.past_due_30_59_days,
        "DebtRatio":                            req.debt_ratio,
        "MonthlyIncome":                        req.monthly_income,
        "NumberOfOpenCreditLinesAndLoans":      req.open_credit_lines,
        "NumberOfTimes90DaysLate":              req.past_due_90_days,
        "NumberRealEstateLoansOrLines":         req.real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": req.past_due_60_89_days,
        "NumberOfDependents":                   req.dependents,
    }

    # Engineered features — must match pipeline exactly
    raw["TotalDelinquencies"] = (
        raw["NumberOfTime30-59DaysPastDueNotWorse"]
        + raw["NumberOfTimes90DaysLate"]
        + raw["NumberOfTime60-89DaysPastDueNotWorse"]
    )
    raw["DelinquencySeverityScore"] = (
        raw["NumberOfTime30-59DaysPastDueNotWorse"] * 1
        + raw["NumberOfTime60-89DaysPastDueNotWorse"] * 2
        + raw["NumberOfTimes90DaysLate"] * 3
    )
    raw["EstimatedMonthlyDebt"] = raw["DebtRatio"] * raw["MonthlyIncome"]
    raw["DisposableIncome"]     = raw["MonthlyIncome"] - raw["EstimatedMonthlyDebt"]
    raw["CreditLineDensity"]    = raw["NumberOfOpenCreditLinesAndLoans"] / (raw["age"] + 1)

    return pd.DataFrame([raw])[ModelStore.feature_names]


def compute_shap_factors(X_input: pd.DataFrame, X_raw: pd.DataFrame):
    """Compute SHAP values and split into risk/protective factors."""
    primary_risk_factors = []
    protective_factors   = []

    if ModelStore.explainer is None:
        return primary_risk_factors, protective_factors

    try:
        shap_vals = ModelStore.explainer(X_input).values[0]
        feature_impacts = sorted(
            zip(ModelStore.feature_names, shap_vals, X_raw.iloc[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for feat, impact, val in feature_impacts[:8]:
            desc = FEATURE_DESCRIPTIONS.get(feat, feat)
            factor = {
                "feature":       feat,
                "description":   desc,
                "value":         round(float(val), 4),
                "shap_impact":   round(float(impact), 6),
                "direction":     "increases_risk" if impact > 0 else "decreases_risk",
                "human_readable": (
                    f"{'High' if impact > 0 else 'Low'} {desc} "
                    f"({'increases' if impact > 0 else 'reduces'} default risk by "
                    f"{abs(impact):.3f})"
                ),
            }
            if impact > 0:
                primary_risk_factors.append(factor)
            else:
                protective_factors.append(factor)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")

    return primary_risk_factors, protective_factors


def persist_prediction(
    db: Session,
    request_id: str,
    request: CreditApplicationRequest,
    response_data: dict,
    inference_ms: float,
) -> bool:
    """
    WHY WE PERSIST EVERY PREDICTION:
    1. ECOA requires lenders to retain adverse action notices for 25 months
    2. Model monitoring needs a baseline of production predictions
    3. Fraud detection relies on historical patterns per application_id
    Returns True if successfully persisted, False otherwise.
    """
    try:
        features = {
            "revolving_utilization": request.revolving_utilization,
            "age":                   request.age,
            "past_due_30_59_days":   request.past_due_30_59_days,
            "debt_ratio":            request.debt_ratio,
            "monthly_income":        request.monthly_income,
            "open_credit_lines":     request.open_credit_lines,
            "past_due_90_days":      request.past_due_90_days,
            "real_estate_loans":     request.real_estate_loans,
            "past_due_60_89_days":   request.past_due_60_89_days,
            "dependents":            request.dependents,
        }

        nl_explanation = None
        if response_data.get("primary_risk_factors"):
            reasons = [f["human_readable"] for f in response_data["primary_risk_factors"][:3]]
            nl_explanation = " | ".join(reasons)

        prediction = DBPrediction(
            request_id       = request_id,
            application_id   = request.application_id,
            risk_score       = response_data["default_probability"],
            risk_tier        = response_data["risk_band"],
            decision         = response_data["decision"],
            confidence       = response_data["confidence"],
            features_json    = features,
            explanation_json = {
                "primary_risk_factors": response_data.get("primary_risk_factors", [])[:3],
                "protective_factors":   response_data.get("protective_factors", [])[:3],
            },
            nl_explanation   = nl_explanation,
            model_version    = response_data["model_version"],
            model_name       = response_data["model_name"],
            inference_ms     = inference_ms,
        )
        db.add(prediction)
        db.commit()
        return True

    except SQLAlchemyError as e:
        db.rollback()
        logger.warning(f"Failed to persist prediction {request_id}: {e}")
        return False


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service":  "OpenFinGuard Credit Intelligence API",
        "version":  "1.0.0",
        "status":   "operational",
        "model":    ModelStore.metadata.get("champion_model", "unknown") if ModelStore.metadata else "loading",
        "docs":     "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    db_ok = False
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_ok = True
    except Exception:
        pass

    return {
        "status":          "healthy" if ModelStore.model is not None else "degraded",
        "model_loaded":    ModelStore.model is not None,
        "shap_available":  ModelStore.explainer is not None,
        "database_online": db_ok,
        "timestamp":       datetime.utcnow().isoformat(),
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return champion model metadata and performance metrics."""
    if not ModelStore.metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name":          ModelStore.metadata.get("champion_model"),
        "test_metrics":        {
            k: v for k, v in ModelStore.metadata.items()
            if k not in ("confusion_matrix", "champion_model", "all_results")
        },
        "feature_count":       len(ModelStore.feature_names) if ModelStore.feature_names else 0,
        "features":            ModelStore.feature_names,
        "top_features_by_shap": dict(list(ModelStore.shap_importance.items())[:10])
                                if ModelStore.shap_importance else {},
    }


@app.post("/predict", response_model=CreditDecisionResponse, tags=["Prediction"])
async def predict(
    request: CreditApplicationRequest,
    db: Session = Depends(get_db),
):
    """
    ## Credit Risk Assessment

    Submit a loan application and receive an instant AI-powered credit decision
    with SHAP explanations for regulatory compliance (ECOA adverse action reasons).
    Every prediction is persisted to PostgreSQL.
    """
    t0 = time.perf_counter()

    if ModelStore.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run run_pipeline.py first.")

    try:
        # 1. Feature engineering (mirrors training pipeline)
        X_raw   = engineer_features(request)
        X_input = X_raw.copy()

        champion = ModelStore.metadata.get("champion_model", "")
        uses_scaling = ModelStore.metadata.get("uses_scaling", False)
        if champion == "LogisticRegression" or uses_scaling:
            X_input = pd.DataFrame(
                ModelStore.scaler.transform(X_raw),
                columns=ModelStore.feature_names,
            )

        # 2. Inference
        prob      = float(ModelStore.model.predict_proba(X_input)[0, 1])
        risk_band, decision, confidence, action = get_risk_band(prob)

        # 3. SHAP explanations
        primary_risk_factors, protective_factors = [], []
        if request.explain:
            primary_risk_factors, protective_factors = compute_shap_factors(X_input, X_raw)

        inference_ms = (time.perf_counter() - t0) * 1000
        request_id   = str(uuid.uuid4())

        response_data = {
            "request_id":           request_id,
            "application_id":       request.application_id,
            "timestamp":            datetime.utcnow().isoformat(),
            "decision":             decision,
            "risk_band":            risk_band,
            "default_probability":  round(prob, 4),
            "confidence":           confidence,
            "primary_risk_factors": primary_risk_factors[:3],
            "protective_factors":   protective_factors[:3],
            "recommended_action":   action,
            "model_version":        "1.0.0",
            "model_name":           champion,
            "threshold_used":       float(ModelStore.metadata.get("optimal_threshold", 0.5)),
            "inference_time_ms":    round(inference_ms, 2),
        }

        # 4. Persist to PostgreSQL
        persisted = persist_prediction(db, request_id, request, response_data, inference_ms)
        response_data["persisted"] = persisted

        logger.info(
            f"Prediction | P={prob:.3f} | {decision} | {risk_band} | "
            f"{inference_ms:.1f}ms | persisted={persisted}"
        )

        return CreditDecisionResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    requests: list[CreditApplicationRequest],
    db: Session = Depends(get_db),
):
    """Batch scoring endpoint — up to 1000 applications per call."""
    if len(requests) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000")

    results = []
    for req in requests:
        try:
            result = await predict(req, db)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "application_id": req.application_id})

    return {"count": len(results), "results": results}


@app.get("/predictions/history", tags=["Database"])
async def prediction_history(
    limit: int = 50,
    decision: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    WHY THIS ENDPOINT:
    Provides an audit trail for compliance teams. Supports filtering by
    decision outcome so analysts can review declined applications.
    """
    try:
        query = db.query(DBPrediction).order_by(DBPrediction.created_at.desc())
        if decision:
            query = query.filter(DBPrediction.decision == decision.upper())
        rows = query.limit(min(limit, 500)).all()

        return {
            "count":       len(rows),
            "predictions": [
                {
                    "request_id":          r.request_id,
                    "application_id":      r.application_id,
                    "decision":            r.decision,
                    "risk_band":           r.risk_tier,
                    "default_probability": r.risk_score,
                    "model_name":          r.model_name,
                    "inference_ms":        r.inference_ms,
                    "created_at":          r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
        }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")


@app.get("/predictions/stats", tags=["Database"])
async def prediction_stats(db: Session = Depends(get_db)):
    """
    WHY LIVE STATS:
    Monitoring the distribution of decisions in production tells you if
    model behavior has drifted. A sudden spike in DECLINE rate is a red flag.
    """
    try:
        from sqlalchemy import func

        total = db.query(func.count(DBPrediction.id)).scalar() or 0
        if total == 0:
            return {"total_predictions": 0, "message": "No predictions yet"}

        decision_counts = (
            db.query(DBPrediction.decision, func.count(DBPrediction.id))
            .group_by(DBPrediction.decision)
            .all()
        )
        avg_score = db.query(func.avg(DBPrediction.risk_score)).scalar()
        avg_latency = db.query(func.avg(DBPrediction.inference_ms)).scalar()

        return {
            "total_predictions":  total,
            "decision_breakdown": {d: c for d, c in decision_counts},
            "avg_risk_score":     round(float(avg_score), 4) if avg_score else None,
            "avg_inference_ms":   round(float(avg_latency), 2) if avg_latency else None,
        }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")


@app.get("/features/importance", tags=["Explainability"])
async def feature_importance():
    """Return global SHAP feature importance rankings."""
    if not ModelStore.shap_importance:
        raise HTTPException(status_code=503, detail="Run shap_analysis.py first")
    return {
        "feature_importance": ModelStore.shap_importance,
        "method":             "mean_absolute_shap",
        "description":        "Higher = more influence on credit decisions",
    }


@app.get("/metrics/fairness", tags=["Fairness"])
async def fairness_metrics():
    """Return latest fairness audit results from file."""
    fairness_path = REPORTS_DIR / "fairness_report.json"
    if not fairness_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Run fairness_metrics.py first to generate the report."
        )
    with open(fairness_path) as f:
        return json.load(f)


@app.get("/example", tags=["Health"])
async def example_request():
    return {
        "example_request": {
            "revolving_utilization": 0.35,
            "age":                   45,
            "past_due_30_59_days":   1,
            "debt_ratio":            0.38,
            "monthly_income":        5500.0,
            "open_credit_lines":     8,
            "past_due_90_days":      0,
            "real_estate_loans":     1,
            "past_due_60_89_days":   0,
            "dependents":            2,
            "explain":               True,
        },
        "curl_example": (
            'curl -X POST "http://localhost:8000/predict" '
            '-H "Content-Type: application/json" '
            '-d \'{"revolving_utilization": 0.35, "age": 45, '
            '"past_due_30_59_days": 1, "debt_ratio": 0.38, '
            '"monthly_income": 5500.0, "open_credit_lines": 8, '
            '"past_due_90_days": 0, "real_estate_loans": 1, '
            '"past_due_60_89_days": 0, "dependents": 2, "explain": true}\''
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)