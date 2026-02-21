"""
OpenFinGuard — Self-Contained Training Pipeline
Runs with: pandas, numpy, scikit-learn, matplotlib, joblib (all standard)
"""

import json
import time
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(".")
DATA_RAW      = ROOT / "data" / "raw"
DATA_PROC     = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports" / "figures"

for d in [DATA_RAW, DATA_PROC, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"
FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

print("=" * 60)
print("OpenFinGuard — Stage 2: Data Pipeline")
print("=" * 60)

# ── 1. Load & Clean ────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/cs-training.csv", index_col=0)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Target distribution:\n{df[TARGET].value_counts(normalize=True).round(3)}")

# Drop duplicate rows
df = df.drop_duplicates()

# Cap extreme outliers
df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
df["DebtRatio"] = df["DebtRatio"].clip(0, 5)
df["age"] = df["age"].clip(18, 100)

# Remove impossible ages
df = df[df["age"] >= 18]

# Fill missing values
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())

print(f"After cleaning: {df.shape[0]:,} rows")
print(f"Missing values remaining: {df[FEATURE_COLS].isnull().sum().sum()}")

# ── 2. Feature Engineering ─────────────────────────────────────────────────
print("\n── Feature Engineering ──")

df["TotalDelinquencies"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"] +
    df["NumberOfTime60-89DaysPastDueNotWorse"] +
    df["NumberOfTimes90DaysLate"]
)

df["DelinquencySeverityScore"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"] * 1 +
    df["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
    df["NumberOfTimes90DaysLate"] * 3
)

df["EstimatedMonthlyDebt"] = df["MonthlyIncome"] * df["DebtRatio"]
df["DisposableIncome"] = df["MonthlyIncome"] - df["EstimatedMonthlyDebt"]
df["CreditLineDensity"] = df["NumberOfOpenCreditLinesAndLoans"] / (df["age"] - 17).clip(1)

ENGINEERED_FEATURES = [
    "TotalDelinquencies", "DelinquencySeverityScore",
    "EstimatedMonthlyDebt", "DisposableIncome", "CreditLineDensity"
]
ALL_FEATURES = FEATURE_COLS + ENGINEERED_FEATURES
print(f"Total features: {len(ALL_FEATURES)} ({len(ENGINEERED_FEATURES)} engineered)")

# ── 3. Train / Val / Test Split ────────────────────────────────────────────
X = df[ALL_FEATURES]
y = df[TARGET]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)
# 0.176 × 0.85 ≈ 0.15 → gives us 70/15/15 split

print(f"\nSplit: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

# ── 4. Handle Class Imbalance (manual oversampling, no imblearn) ────────────
print("\n── Handling Class Imbalance ──")
train_df = pd.concat([X_train, y_train], axis=1)
majority = train_df[train_df[TARGET] == 0]
minority = train_df[train_df[TARGET] == 1]

minority_upsampled = resample(minority, replace=True, n_samples=len(majority) // 3, random_state=42)
train_balanced = pd.concat([majority, minority_upsampled])

X_train_bal = train_balanced[ALL_FEATURES]
y_train_bal = train_balanced[TARGET]
print(f"Balanced train: {y_train_bal.value_counts().to_dict()}")

# ── 5. Scale features ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=ALL_FEATURES)
X_val_scaled   = pd.DataFrame(scaler.transform(X_val),  columns=ALL_FEATURES)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=ALL_FEATURES)

# Save processed data
X_val.to_csv(DATA_PROC / "X_val.csv", index=False)
y_val.to_csv(DATA_PROC / "y_val.csv", index=False)
X_test.to_csv(DATA_PROC / "X_test.csv", index=False)
y_test.to_csv(DATA_PROC / "y_test.csv", index=False)
print(f"Processed data saved → {DATA_PROC}")

# ── 6. KS Statistic ───────────────────────────────────────────────────────
def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

# ── 7. Train All Models ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OpenFinGuard — Stage 3: Model Training & Comparison")
print("=" * 60)

MODELS = {
    "LogisticRegression": (
        LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
        True  # needs scaling
    ),
    "DecisionTree": (
        DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                               class_weight="balanced", random_state=42),
        False
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=30,
                               class_weight="balanced", n_jobs=-1, random_state=42),
        False
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, random_state=42),
        False
    ),
}

results = {}
trained_models = {}

for name, (model, use_scaled) in MODELS.items():
    print(f"\n▶ Training {name}...")
    X_tr = X_train_scaled if use_scaled else X_train_bal
    X_v  = X_val_scaled   if use_scaled else X_val

    t0 = time.time()
    model.fit(X_tr, y_train_bal)
    train_time = time.time() - t0

    # Inference time (1000 samples)
    t0 = time.time()
    for _ in range(10):
        model.predict_proba(X_v.iloc[:100])
    infer_ms = (time.time() - t0) / 10 * 10  # ms per 100 samples

    y_prob = model.predict_proba(X_v)[:, 1]

    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    opt_idx = np.argmax(f1_scores[:-1])
    opt_threshold = float(thresholds[opt_idx])
    y_pred = (y_prob >= opt_threshold).astype(int)

    auc   = roc_auc_score(y_val, y_prob)
    ks    = ks_statistic(y_val, y_prob)
    prauc = average_precision_score(y_val, y_prob)
    f1    = f1_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_prob)

    results[name] = {
        "AUC-ROC": round(auc, 4),
        "KS-Stat": round(ks, 4),
        "PR-AUC":  round(prauc, 4),
        "F1":      round(f1, 4),
        "Brier":   round(brier, 4),
        "TrainTime_s": round(train_time, 1),
        "InferTime_ms": round(infer_ms, 2),
        "OptThreshold": round(opt_threshold, 4),
    }
    trained_models[name] = (model, use_scaled)

    print(f"  AUC={auc:.4f} | KS={ks:.4f} | PR-AUC={prauc:.4f} | F1={f1:.4f} | Brier={brier:.4f}")
    print(f"  Train: {train_time:.1f}s | Infer: {infer_ms:.1f}ms | Threshold: {opt_threshold:.3f}")

# ── 8. Compare & Select Champion ──────────────────────────────────────────
print("\n── Model Comparison Table ──")
results_df = pd.DataFrame(results).T
print(results_df.to_string())

# Composite score: weighted blend of key metrics
# AUC (40%) + KS (30%) + PR-AUC (20%) + penalty for slow inference (10%)
max_infer = results_df["InferTime_ms"].max()
results_df["CompositeScore"] = (
    0.40 * results_df["AUC-ROC"] +
    0.30 * results_df["KS-Stat"] +
    0.20 * results_df["PR-AUC"] +
    0.10 * (1 - results_df["InferTime_ms"] / max_infer)
)

champion_name = results_df["CompositeScore"].idxmax()
print(f"\n🏆 Champion: {champion_name} (composite score: {results_df.loc[champion_name, 'CompositeScore']:.4f})")

# ── 9. Save Champion ───────────────────────────────────────────────────────
champion_model, champion_scaled = trained_models[champion_name]
joblib.dump(champion_model, MODELS_DIR / "champion_model.joblib")
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

with open(MODELS_DIR / "feature_names.json", "w") as f:
    json.dump(ALL_FEATURES, f)

metadata = {
    "champion_model": champion_name,
    "optimal_threshold": results[champion_name]["OptThreshold"],
    "val_auc": results[champion_name]["AUC-ROC"],
    "val_ks": results[champion_name]["KS-Stat"],
    "val_prauc": results[champion_name]["PR-AUC"],
    "val_f1": results[champion_name]["F1"],
    "uses_scaling": champion_scaled,
    "all_results": results,
}
with open(MODELS_DIR / "champion_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

results_df.to_csv(MODELS_DIR / "model_comparison.csv")
print(f"Champion saved → {MODELS_DIR}")

# ── 10. Generate Comparison Plot ──────────────────────────────────────────
print("\n── Generating Plots ──")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("OpenFinGuard — Model Comparison Dashboard", fontsize=15, fontweight="bold")

metrics_to_plot = ["AUC-ROC", "KS-Stat", "PR-AUC", "F1", "Brier", "CompositeScore"]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
model_names = list(results.keys())

for ax, metric in zip(axes.flat, metrics_to_plot):
    vals = [results_df.loc[m, metric] for m in model_names]
    bars = ax.bar(model_names, vals, color=colors)
    ax.set_title(metric, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.tick_params(axis="x", rotation=30)
    # Highlight champion
    champ_idx = model_names.index(champion_name)
    bars[champ_idx].set_edgecolor("gold")
    bars[champ_idx].set_linewidth(3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 11. ROC Curves ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("OpenFinGuard — ROC & Precision-Recall Curves", fontweight="bold")

for (name, (model, use_scaled)), color in zip(trained_models.items(), colors):
    X_v = X_val_scaled if use_scaled else X_val
    y_prob = model.predict_proba(X_v)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc = results[name]["AUC-ROC"]
    ax1.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
             lw=2 if name == champion_name else 1.2,
             color=color,
             linestyle="-" if name == champion_name else "--")

    prec, rec, _ = precision_recall_curve(y_val, y_prob)
    prauc = results[name]["PR-AUC"]
    ax2.plot(rec, prec, label=f"{name} (PR-AUC={prauc:.3f})",
             lw=2 if name == champion_name else 1.2,
             color=color,
             linestyle="-" if name == champion_name else "--")

ax1.plot([0,1],[0,1],"k--", lw=0.8)
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curves"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 12. SHAP-style Feature Importance ─────────────────────────────────────
print("── Feature Importance ──")
# Get feature importances from the champion (or best tree model)
best_tree = None
for name in ["GradientBoosting", "RandomForest", "DecisionTree"]:
    if name in trained_models:
        best_tree = trained_models[name][0]
        best_tree_name = name
        break

if best_tree is not None and hasattr(best_tree, "feature_importances_"):
    importances = pd.Series(best_tree.feature_importances_, index=ALL_FEATURES).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_bar = ["#d62728" if imp > importances.median() else "#1f77b4"
                  for imp in importances]
    ax.barh(importances.index, importances.values, color=colors_bar)
    ax.set_title(f"Feature Importances — {best_tree_name} (Champion Proxy)",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot saved")

    # Save as JSON (replaces shap_feature_importance for API use)
    importances.sort_values(ascending=False).to_json(
        MODELS_DIR / "shap_feature_importance.json"
    )

# ── 13. Calibration Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
champ_model, champ_scaled = trained_models[champion_name]
X_v = X_val_scaled if champ_scaled else X_val
y_prob_champ = champ_model.predict_proba(X_v)[:, 1]

fraction_pos, mean_pred = calibration_curve(y_val, y_prob_champ, n_bins=10)
ax.plot(mean_pred, fraction_pos, "s-", label=champion_name, color="#2196F3")
ax.plot([0,1],[0,1],"k--", label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title(f"Calibration Curve — {champion_name}", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 14. Test Set Evaluation ───────────────────────────────────────────────
print("\n── Final Test Set Evaluation ──")
X_t = X_test_scaled if champ_scaled else X_test
y_prob_test = champ_model.predict_proba(X_t)[:, 1]
threshold = results[champion_name]["OptThreshold"]
y_pred_test = (y_prob_test >= threshold).astype(int)

test_auc   = roc_auc_score(y_test, y_prob_test)
test_ks    = ks_statistic(y_test, y_prob_test)
test_prauc = average_precision_score(y_test, y_prob_test)
test_f1    = f1_score(y_test, y_pred_test)
test_brier = brier_score_loss(y_test, y_prob_test)

print(f"Test AUC={test_auc:.4f} | KS={test_ks:.4f} | PR-AUC={test_prauc:.4f} | F1={test_f1:.4f}")

# Update metadata with test scores
metadata["test_auc"]   = round(test_auc, 4)
metadata["test_ks"]    = round(test_ks, 4)
metadata["test_prauc"] = round(test_prauc, 4)
metadata["test_f1"]    = round(test_f1, 4)
with open(MODELS_DIR / "champion_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ── 15. Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted: Paid", "Predicted: Default"],
            yticklabels=["Actual: Paid", "Actual: Default"])
ax.set_title(f"Confusion Matrix — {champion_name}\n(Test Set, threshold={threshold:.2f})",
             fontweight="bold")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("✅ Pipeline Complete!")
print("=" * 60)
print(f"\n🏆 Champion: {champion_name}")
print(f"   Val  AUC={results[champion_name]['AUC-ROC']:.4f} | KS={results[champion_name]['KS-Stat']:.4f}")
print(f"   Test AUC={test_auc:.4f} | KS={test_ks:.4f} | F1={test_f1:.4f}")
print(f"\n📁 Models saved to: {MODELS_DIR}")
print(f"📊 Plots saved to:  {REPORTS_DIR}")
print(f"\nFiles generated:")
for f in sorted(list(MODELS_DIR.glob("*")) + list(REPORTS_DIR.glob("*"))):
    print(f"  {f.relative_to(ROOT)}")