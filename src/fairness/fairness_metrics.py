"""
OpenFinGuard — Fairness & Bias Analysis
=========================================
WHY THIS FILE EXISTS:
  Credit scoring is regulated under ECOA (Equal Credit Opportunity Act) and
  the Fair Housing Act. A model that discriminates by age, race, or gender —
  even unintentionally — exposes lenders to massive legal liability.

  We measure:
  1. Demographic Parity — equal approval rates across groups
  2. Equal Opportunity — equal TPR (sensitivity) across groups
  3. Predictive Parity — equal precision across groups
  4. KS disparity — equal separation ability across groups

  This is what Microsoft Research's Fairlearn team works on. This module
  demonstrates you understand responsible AI beyond theory.
"""

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Fairness Thresholds ────────────────────────────────────────────────────
# WHY THESE THRESHOLDS:
# The "80% rule" (0.8) is the EEOC's four-fifths rule — the legal standard
# for disparate impact in the US. Anything below 0.8 is legally concerning.
DEMOGRAPHIC_PARITY_THRESHOLD = 0.80
EQUAL_OPPORTUNITY_THRESHOLD  = 0.80
PREDICTIVE_PARITY_THRESHOLD  = 0.80


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-group fairness metrics.

    Metrics explained:
    - Approval Rate: % predicted non-default (positive outcome for borrower)
    - TPR (Recall): Of true defaulters, how many did we catch? (Equal Opportunity)
    - TNR (Specificity): Of true non-defaulters, how many did we correctly approve?
    - Precision: Of predicted defaulters, how many truly defaulted? (Predictive Parity)
    - AUC: Discrimination ability per group
    - Selection Rate: % approved (Demographic Parity numerator)
    """
    rows = []
    groups = sorted(np.unique(group_labels))

    for group in groups:
        mask = group_labels == group
        yt, yp, yprob = y_true[mask], y_pred[mask], y_prob[mask]

        if len(yt) < 10 or yt.sum() < 2:
            continue  # skip groups too small for reliable metrics

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        n = len(yt)

        rows.append({
            "group": str(group),
            "n_samples": n,
            "n_defaults": int(yt.sum()),
            "default_rate": round(float(yt.mean()), 4),
            "approval_rate": round(float((yp == 0).mean()), 4),  # predicted non-default
            "tpr_recall": round(float(tp / (tp + fn + 1e-9)), 4),
            "tnr_specificity": round(float(tn / (tn + fp + 1e-9)), 4),
            "fpr": round(float(fp / (fp + tn + 1e-9)), 4),
            "fnr": round(float(fn / (fn + tp + 1e-9)), 4),
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "auc_roc": round(float(roc_auc_score(yt, yprob)), 4),
        })

    return pd.DataFrame(rows)


def compute_fairness_ratios(group_df: pd.DataFrame, reference_group: str = None) -> pd.DataFrame:
    """
    Compute fairness ratios relative to the reference (most favorable) group.
    WHY RATIOS: Absolute differences are hard to compare; ratios give the
    proportion by which protected groups are disadvantaged.
    """
    if reference_group is None:
        # Reference = group with highest approval rate (most favorable)
        reference_group = group_df.loc[group_df["approval_rate"].idxmax(), "group"]

    ref_row = group_df[group_df["group"] == reference_group].iloc[0]

    ratios = []
    for _, row in group_df.iterrows():
        ratios.append({
            "group": row["group"],
            "reference_group": reference_group,
            "demographic_parity_ratio": round(row["approval_rate"] / (ref_row["approval_rate"] + 1e-9), 4),
            "equal_opportunity_ratio": round(row["tpr_recall"] / (ref_row["tpr_recall"] + 1e-9), 4),
            "predictive_parity_ratio": round(row["precision"] / (ref_row["precision"] + 1e-9), 4),
            "auc_ratio": round(row["auc_roc"] / (ref_row["auc_roc"] + 1e-9), 4),
            # Flag violations
            "demographic_parity_violation": row["approval_rate"] / (ref_row["approval_rate"] + 1e-9) < DEMOGRAPHIC_PARITY_THRESHOLD,
            "equal_opportunity_violation": row["tpr_recall"] / (ref_row["tpr_recall"] + 1e-9) < EQUAL_OPPORTUNITY_THRESHOLD,
            "predictive_parity_violation": row["precision"] / (ref_row["precision"] + 1e-9) < PREDICTIVE_PARITY_THRESHOLD,
        })

    return pd.DataFrame(ratios)


def plot_fairness_dashboard(
    group_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
    attribute_name: str,
) -> None:
    """Comprehensive fairness visualization dashboard."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"OpenFinGuard — Fairness Analysis by {attribute_name}",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = plt.cm.Set2.colors
    groups = group_df["group"].tolist()
    group_colors = {g: colors[i % len(colors)] for i, g in enumerate(groups)}

    # 1. Approval rates
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(group_df["group"], group_df["approval_rate"],
                   color=[group_colors[g] for g in group_df["group"]])
    ax1.axhline(group_df["approval_rate"].max() * DEMOGRAPHIC_PARITY_THRESHOLD,
                color="red", linestyle="--", alpha=0.7, label="80% rule threshold")
    ax1.set_title("Approval Rate by Group\n(Demographic Parity)")
    ax1.set_ylabel("Approval Rate")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 1)

    # 2. TPR (Equal Opportunity)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(group_df["group"], group_df["tpr_recall"],
            color=[group_colors[g] for g in group_df["group"]])
    ax2.axhline(group_df["tpr_recall"].max() * EQUAL_OPPORTUNITY_THRESHOLD,
                color="red", linestyle="--", alpha=0.7)
    ax2.set_title("True Positive Rate by Group\n(Equal Opportunity)")
    ax2.set_ylabel("TPR / Recall")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 1)

    # 3. AUC by group
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(group_df["group"], group_df["auc_roc"],
            color=[group_colors[g] for g in group_df["group"]])
    ax3.set_title("AUC-ROC by Group\n(Predictive Performance Parity)")
    ax3.set_ylabel("AUC-ROC")
    ax3.tick_params(axis="x", rotation=30)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(0.5, 1.0)

    # 4. Fairness ratio heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    ratio_cols = ["demographic_parity_ratio", "equal_opportunity_ratio",
                  "predictive_parity_ratio", "auc_ratio"]
    heat_data = ratio_df.set_index("group")[ratio_cols]

    # Color: green near 1.0, red near 0.8 threshold
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap=cmap, center=1.0,
                vmin=0.6, vmax=1.2, ax=ax4, linewidths=0.5, cbar=True)
    ax4.set_title("Fairness Ratios (1.0 = perfect parity | red line = 0.8 EEOC threshold)")
    ax4.set_xticklabels(["Demographic\nParity", "Equal\nOpportunity",
                         "Predictive\nParity", "AUC\nParity"], rotation=0)

    # 5. Violation flags
    ax5 = fig.add_subplot(gs[1, 2])
    violation_cols = ["demographic_parity_violation", "equal_opportunity_violation",
                      "predictive_parity_violation"]
    violation_data = ratio_df.set_index("group")[violation_cols].astype(int)
    sns.heatmap(violation_data, annot=True, fmt="d", cmap=["#2ca02c", "#d62728"],
                vmin=0, vmax=1, ax=ax5, cbar=False, linewidths=0.5)
    ax5.set_title("Violations (1=violation, 0=OK)")
    ax5.set_xticklabels(["DemParity", "EqOpp", "PredParity"], rotation=30)

    # 6. Default rate vs approval rate scatter
    ax6 = fig.add_subplot(gs[2, 0])
    for _, row in group_df.iterrows():
        ax6.scatter(row["default_rate"], row["approval_rate"],
                    s=row["n_samples"] / 50, alpha=0.8,
                    color=group_colors[row["group"]], label=row["group"])
        ax6.annotate(row["group"], (row["default_rate"], row["approval_rate"]),
                     fontsize=8, ha="left")
    ax6.set_xlabel("Actual Default Rate")
    ax6.set_ylabel("Model Approval Rate")
    ax6.set_title("Default Rate vs Approval Rate\n(bubble size = group size)")
    ax6.grid(True, alpha=0.3)

    # 7. FPR and FNR by group
    ax7 = fig.add_subplot(gs[2, 1])
    x = np.arange(len(group_df))
    width = 0.35
    ax7.bar(x - width/2, group_df["fpr"], width, label="FPR (false alarms)", alpha=0.8)
    ax7.bar(x + width/2, group_df["fnr"], width, label="FNR (missed defaults)", alpha=0.8)
    ax7.set_xticks(x)
    ax7.set_xticklabels(group_df["group"], rotation=30)
    ax7.set_title("False Positive & Negative Rates\nby Group")
    ax7.set_ylabel("Rate")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis="y")

    # 8. Sample sizes
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.bar(group_df["group"], group_df["n_samples"],
            color=[group_colors[g] for g in group_df["group"]])
    ax8.set_title("Sample Size by Group\n(reliability check)")
    ax8.set_ylabel("N samples")
    ax8.tick_params(axis="x", rotation=30)
    ax8.grid(True, alpha=0.3, axis="y")

    out_path = REPORTS_DIR / f"fairness_{attribute_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Fairness dashboard saved → {out_path}")


def generate_fairness_report(
    group_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
    attribute_name: str,
) -> dict:
    """Generate structured fairness report for the responsible AI document."""
    violations = ratio_df[
        ratio_df["demographic_parity_violation"] |
        ratio_df["equal_opportunity_violation"] |
        ratio_df["predictive_parity_violation"]
    ]

    report = {
        "attribute": attribute_name,
        "reference_group": ratio_df["reference_group"].iloc[0],
        "n_groups": len(group_df),
        "violations_found": len(violations) > 0,
        "violation_count": len(violations),
        "violating_groups": violations["group"].tolist(),
        "overall_summary": {
            "min_approval_rate": float(group_df["approval_rate"].min()),
            "max_approval_rate": float(group_df["approval_rate"].max()),
            "approval_rate_range": float(group_df["approval_rate"].max() - group_df["approval_rate"].min()),
            "min_tpr": float(group_df["tpr_recall"].min()),
            "max_tpr": float(group_df["tpr_recall"].max()),
            "min_auc": float(group_df["auc_roc"].min()),
            "max_auc": float(group_df["auc_roc"].max()),
        },
        "group_metrics": group_df.to_dict(orient="records"),
        "fairness_ratios": ratio_df.to_dict(orient="records"),
    }

    status = "⚠️ VIOLATIONS DETECTED" if report["violations_found"] else "✅ PASSES FAIRNESS CHECKS"
    logger.info(f"\nFairness Report — {attribute_name}: {status}")
    if report["violations_found"]:
        logger.warning(f"  Violating groups: {report['violating_groups']}")
    else:
        logger.success(f"  All groups pass the 80% rule threshold")

    return report


def run_fairness_analysis() -> dict:
    """Full fairness analysis pipeline."""
    logger.info("=" * 60)
    logger.info("OpenFinGuard — Fairness & Bias Analysis")
    logger.info("=" * 60)

    # Load artifacts
    model    = joblib.load(MODELS_DIR / "champion_model.joblib")
    scaler   = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "champion_metadata.json") as f:
        metadata = json.load(f)
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    threshold = metadata["optimal_threshold"]
    champion  = metadata["champion_model"]

    X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze()

    X_input = X_val[feature_names]
    if champion == "LogisticRegression":
        X_input = pd.DataFrame(scaler.transform(X_input), columns=feature_names)

    y_prob = model.predict_proba(X_input)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    y_true_arr = y_val.values
    X_val_reset = X_val.reset_index(drop=True)

    all_reports = {}

    # ── Fairness by Age Group ──────────────────────────────────────────────
    logger.info("\n── Analysis 1: Fairness by Age Group ──")
    age_groups = pd.cut(
        X_val_reset["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
    ).astype(str)

    group_df_age = compute_group_metrics(y_true_arr, y_pred, y_prob, age_groups.values)
    ratio_df_age = compute_fairness_ratios(group_df_age)
    plot_fairness_dashboard(group_df_age, ratio_df_age, "Age Group")
    all_reports["age_group"] = generate_fairness_report(group_df_age, ratio_df_age, "Age Group")

    # ── Fairness by Income Quartile ────────────────────────────────────────
    logger.info("\n── Analysis 2: Fairness by Income Quartile ──")
    income_quartiles = pd.qcut(
        X_val_reset["MonthlyIncome"],
        q=4,
        labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"],
    ).astype(str)

    group_df_inc = compute_group_metrics(y_true_arr, y_pred, y_prob, income_quartiles.values)
    ratio_df_inc = compute_fairness_ratios(group_df_inc)
    plot_fairness_dashboard(group_df_inc, ratio_df_inc, "Income Quartile")
    all_reports["income_quartile"] = generate_fairness_report(group_df_inc, ratio_df_inc, "Income Quartile")

    # ── Fairness by Dependents ─────────────────────────────────────────────
    logger.info("\n── Analysis 3: Fairness by Number of Dependents ──")
    dep_groups = X_val_reset["NumberOfDependents"].apply(
        lambda x: "0" if x == 0 else "1-2" if x <= 2 else "3+"
    )

    group_df_dep = compute_group_metrics(y_true_arr, y_pred, y_prob, dep_groups.values)
    ratio_df_dep = compute_fairness_ratios(group_df_dep)
    plot_fairness_dashboard(group_df_dep, ratio_df_dep, "Dependents")
    all_reports["dependents"] = generate_fairness_report(group_df_dep, ratio_df_dep, "Dependents")

    # ── Save full report ───────────────────────────────────────────────────
    with open(REPORTS_DIR / "fairness_report.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)

    logger.success(f"Full fairness report → {REPORTS_DIR / 'fairness_report.json'}")
    logger.success("Stage 5 complete. Ready for FastAPI (Stage 6).")

    return all_reports


if __name__ == "__main__":
    run_fairness_analysis()