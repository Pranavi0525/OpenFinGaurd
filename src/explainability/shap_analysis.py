"""
OpenFinGuard — SHAP Explainability
===================================
WHY THIS FILE EXISTS:
  Predictions without explanations are black boxes. Regulators (ECOA, FCRA) require
  lenders to provide "adverse action reasons" — you cannot legally say "the model
  said no." You must say WHY. SHAP gives us that.

  Microsoft Research differentiator: Most projects compute SHAP. We go further:
  - Global feature importance (what matters overall)
  - Local explanations (why THIS borrower was denied)
  - Natural language reasons (what a loan officer would say)
  - Fairness-aware SHAP (do explanations differ by demographic group?)
"""

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Human-readable feature descriptions ───────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "RevolvingUtilizationOfUnsecuredLines": "credit card utilization ratio",
    "age": "borrower age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59 day late payments",
    "DebtRatio": "debt-to-income ratio",
    "MonthlyIncome": "monthly income",
    "NumberOfOpenCreditLinesAndLoans": "open credit accounts",
    "NumberOfTimes90DaysLate": "90+ day late payments",
    "NumberRealEstateLoansOrLines": "real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 day late payments",
    "NumberOfDependents": "number of dependents",
    "TotalDelinquencies": "total delinquency count",
    "DelinquencySeverityScore": "weighted delinquency severity",
    "EstimatedMonthlyDebt": "estimated monthly debt payment",
    "DisposableIncome": "estimated disposable income",
    "CreditLineDensity": "credit lines per year of age",
}

# ── Risk thresholds (calibrated from training data) ────────────────────────
RISK_BANDS = [
    (0.00, 0.05, "Very Low Risk",   "Approve — Excellent credit profile"),
    (0.05, 0.15, "Low Risk",        "Approve — Good credit profile"),
    (0.15, 0.30, "Moderate Risk",   "Review — Standard underwriting required"),
    (0.30, 0.50, "High Risk",       "Review — Enhanced underwriting required"),
    (0.50, 1.00, "Very High Risk",  "Decline — Does not meet credit criteria"),
]


def get_risk_band(probability: float) -> tuple:
    for lo, hi, label, action in RISK_BANDS:
        if lo <= probability < hi:
            return label, action
    return "Very High Risk", "Decline — Does not meet credit criteria"


def load_artifacts():
    """Load champion model, scaler, and validation data."""
    model   = joblib.load(MODELS_DIR / "champion_model.joblib")
    scaler  = joblib.load(MODELS_DIR / "scaler.joblib")

    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(MODELS_DIR / "champion_metadata.json") as f:
        metadata = json.load(f)

    X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze()

    return model, scaler, feature_names, metadata, X_val, y_val


# ── SHAP Analysis ──────────────────────────────────────────────────────────
def compute_shap_values(model, X_sample: pd.DataFrame) -> shap.Explanation:
    """
    WHY TREESHAP:
    TreeSHAP is an exact SHAP algorithm for tree-based models (LightGBM, XGBoost,
    RandomForest). It runs in O(TLD²) time — fast enough for real-time API use.
    For linear models, we'd use LinearExplainer instead.
    """
    model_type = type(model).__name__
    logger.info(f"Computing SHAP values for {model_type} on {len(X_sample):,} samples...")

    try:
        # Tree-based models — fast exact SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        logger.success(f"SHAP values computed: shape={shap_values.values.shape}")
        return explainer, shap_values
    except Exception:
        # Fallback for non-tree models (LogisticRegression)
        logger.info("Falling back to KernelExplainer (slower, model-agnostic)...")
        background = shap.sample(X_sample, 100, random_state=42)
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        shap_values_raw = explainer.shap_values(X_sample[:200])
        return explainer, shap_values_raw


def plot_global_shap(shap_values, X_sample: pd.DataFrame) -> None:
    """
    Global SHAP: What features matter most ACROSS all borrowers?
    WHY: This is your model's "reasoning fingerprint" — it shows whether the model
    learned real credit risk signals or spurious correlations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("OpenFinGuard — Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")

    # Bar plot — mean absolute SHAP
    plt.sca(axes[0])
    shap.plots.bar(shap_values, max_display=15, show=False)
    axes[0].set_title("Mean |SHAP Value| — Overall Impact")

    # Beeswarm — direction + magnitude
    plt.sca(axes[1])
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    axes[1].set_title("SHAP Beeswarm — Impact Direction & Magnitude\n"
                      "(red = high feature value, blue = low)")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_global.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Global SHAP plot saved → {REPORTS_DIR / 'shap_global.png'}")


def plot_shap_interactions(shap_values, X_sample: pd.DataFrame) -> None:
    """
    WHY INTERACTION PLOTS:
    SHAP dependency plots reveal HOW a feature affects risk — not just how much.
    E.g.: Utilization is harmless at 10% but catastrophic at 90%.
    This is the kind of insight that separates ML from statistics.
    """
    top_features = (
        pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_sample.columns)
        .nlargest(4)
        .index.tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("SHAP Dependency Plots — Top 4 Features", fontsize=13, fontweight="bold")

    for ax, feature in zip(axes.flat, top_features):
        idx = list(X_sample.columns).index(feature)
        shap.plots.scatter(
            shap_values[:, feature],
            color=shap_values[:, top_features[0]],
            ax=ax,
            show=False,
        )
        ax.set_title(f"{FEATURE_DESCRIPTIONS.get(feature, feature)}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_interactions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP interaction plots saved → {REPORTS_DIR / 'shap_interactions.png'}")


def explain_single_prediction(
    model,
    explainer,
    X_sample: pd.DataFrame,
    idx: int,
    threshold: float,
) -> dict:
    """
    Local SHAP: Why was THIS specific borrower approved/denied?

    WHY THIS MATTERS:
    ECOA (Equal Credit Opportunity Act) requires adverse action notices.
    This function generates the legally required explanation for every decision.
    It's not just good engineering — it's regulatory compliance.
    """
    borrower = X_sample.iloc[[idx]]
    prob = model.predict_proba(borrower)[0, 1]
    decision = "DECLINE" if prob >= threshold else "APPROVE"
    risk_label, action = get_risk_band(prob)

    # SHAP for this individual
    shap_vals = explainer(borrower).values[0]
    feature_impacts = pd.Series(shap_vals, index=X_sample.columns).sort_values(key=abs, ascending=False)

    # Natural language reasons
    reasons = []
    for feat, impact in feature_impacts.head(5).items():
        val = borrower[feat].iloc[0]
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        direction = "increased" if impact > 0 else "decreased"
        reasons.append(
            f"{'⬆' if impact > 0 else '⬇'} Your {desc} ({val:.2f}) "
            f"{direction} risk by {abs(impact):.4f}"
        )

    explanation = {
        "borrower_index": idx,
        "default_probability": round(float(prob), 4),
        "decision": decision,
        "risk_band": risk_label,
        "recommended_action": action,
        "top_risk_factors": reasons,
        "shap_values": {
            feat: round(float(val), 6)
            for feat, val in feature_impacts.head(10).items()
        },
        "feature_values": borrower.iloc[0].to_dict(),
    }

    return explanation


def plot_local_explanation(
    model,
    explainer,
    X_sample: pd.DataFrame,
    idx: int,
    threshold: float,
) -> None:
    """Waterfall plot for a single borrower — the 'why' visualization."""
    borrower = X_sample.iloc[[idx]]
    prob = model.predict_proba(borrower)[0, 1]
    decision = "DECLINE" if prob >= threshold else "APPROVE"
    color = "#d62728" if decision == "DECLINE" else "#2ca02c"

    shap_explanation = explainer(borrower)

    fig, ax = plt.subplots(figsize=(12, 7))
    shap.plots.waterfall(shap_explanation[0], max_display=12, show=False)
    ax = plt.gca()
    ax.set_title(
        f"Local Explanation — Borrower #{idx}\n"
        f"Default Probability: {prob:.1%} | Decision: {decision}",
        fontsize=12, fontweight="bold", color=color,
    )

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"shap_local_borrower_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Local explanation plot → {REPORTS_DIR / f'shap_local_borrower_{idx}.png'}")


def analyze_shap_by_age_group(
    model,
    explainer,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """
    WHY FAIRNESS-AWARE SHAP:
    Do we explain decisions differently for young vs old borrowers?
    If age-correlated features (DebtRatio, CreditLineDensity) have very different
    SHAP distributions across age groups, the model may be using age as a proxy.
    This is disparate impact — illegal in credit decisions in many jurisdictions.
    """
    age_bins = pd.cut(X_val["age"], bins=[0, 30, 40, 50, 60, 100],
                      labels=["18-30", "31-40", "41-50", "51-60", "60+"])

    shap_vals = explainer(X_val).values  # shape: (n, features)
    shap_df = pd.DataFrame(shap_vals, columns=X_val.columns)
    shap_df["age_group"] = age_bins.values

    # Mean |SHAP| per feature per age group
    group_importance = (
        shap_df.groupby("age_group")[list(X_val.columns)]
        .apply(lambda g: g.abs().mean())
    )

    logger.info("\nSHAP Feature Importance by Age Group:")
    print(group_importance.round(4).to_string())

    # Plot
    top_feats = shap_df[list(X_val.columns)].abs().mean().nlargest(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SHAP Feature Importance by Age Group\n(Fairness-Aware Analysis)",
                 fontsize=13, fontweight="bold")

    for ax, feat in zip(axes.flat, top_feats):
        group_importance[feat].plot(kind="bar", ax=ax, color=plt.cm.Set2.colors[:5])
        ax.set_title(FEATURE_DESCRIPTIONS.get(feat, feat))
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Mean |SHAP|")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_fairness_by_age.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Fairness SHAP plot → {REPORTS_DIR / 'shap_fairness_by_age.png'}")

    return group_importance


def run_shap_analysis() -> None:
    """Full SHAP pipeline."""
    logger.info("=" * 60)
    logger.info("OpenFinGuard — SHAP Explainability Analysis")
    logger.info("=" * 60)

    model, scaler, feature_names, metadata, X_val, y_val = load_artifacts()
    threshold = metadata["optimal_threshold"]

    champion_name = metadata["champion_model"]
    use_scaled = champion_name == "LogisticRegression"

    X_sample = X_val[feature_names]
    if use_scaled:
        X_sample = pd.DataFrame(scaler.transform(X_sample), columns=feature_names)

    # Use a representative sample for global SHAP (speed vs completeness tradeoff)
    sample_size = min(2000, len(X_sample))
    X_shap = X_sample.sample(sample_size, random_state=42).reset_index(drop=True)

    # Compute SHAP
    explainer, shap_values = compute_shap_values(model, X_shap)

    # Global analysis
    logger.info("\n── Global SHAP Analysis ──")
    plot_global_shap(shap_values, X_shap)
    plot_shap_interactions(shap_values, X_shap)

    # Local explanations — one high-risk, one low-risk borrower
    logger.info("\n── Local Explanations ──")
    probs = model.predict_proba(X_shap)[:, 1]

    high_risk_idx = int(np.argmax(probs))
    low_risk_idx  = int(np.argmin(probs))

    for idx in [high_risk_idx, low_risk_idx]:
        explanation = explain_single_prediction(model, explainer, X_shap, idx, threshold)
        logger.info(
            f"\nBorrower #{idx} | P(default)={explanation['default_probability']:.1%} "
            f"| {explanation['decision']} | {explanation['risk_band']}"
        )
        for reason in explanation["top_risk_factors"]:
            logger.info(f"  {reason}")
        plot_local_explanation(model, explainer, X_shap, idx, threshold)

    # Fairness-aware SHAP
    logger.info("\n── Fairness-Aware SHAP Analysis ──")
    analyze_shap_by_age_group(model, explainer, X_shap, y_val.iloc[:sample_size])

    # Save SHAP values for API use
    mean_shap = (
        pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_shap.columns)
        .sort_values(ascending=False)
    )
    mean_shap.to_json(MODELS_DIR / "shap_feature_importance.json")
    logger.success(f"SHAP feature importance saved → {MODELS_DIR / 'shap_feature_importance.json'}")

    logger.success("\nStage 4 complete. Ready for Fairness Metrics (Stage 5).")


if __name__ == "__main__":
    run_shap_analysis()