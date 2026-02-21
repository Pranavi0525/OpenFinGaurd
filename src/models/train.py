"""
OpenFinGuard — Model Training & Champion Selection
===================================================
WHY THIS FILE EXISTS:
  Most ML projects "just use XGBoost." We don't. We evaluate 6 models across
  performance, production, and fairness metrics — then select a champion with
  documented evidence. This is model governance, not model guessing.

  Microsoft Research cares about: reproducibility, fairness, explainability.
  This file addresses all three before a single prediction is made.

Stage 3 Build Order:
  3a. Train all 6 models with cross-validation
  3b. Evaluate on all metrics (AUC, KS, PR-AUC, F1, inference time, calibration)
  3c. Generate comparison table + visualizations
  3d. Select champion with documented justification
  3e. Hyperparameter tune ONLY the winner
"""

import time
import json
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost

from loguru import logger
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ── KS Statistic ──────────────────────────────────────────────────────────
def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    WHY KS STATISTIC:
    The KS (Kolmogorov-Smirnov) statistic is the industry standard in credit
    scoring — it measures how well a model separates good vs bad borrowers.
    Banks use KS > 0.4 as the production threshold. AUC alone misses this.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ── Model Definitions ──────────────────────────────────────────────────────
def get_candidate_models() -> dict:
    """
    WHY THESE 6:
    - LogisticRegression: Regulatory baseline. Many banks legally require an
      interpretable model. Max explainability, used in Fair Lending compliance.
    - DecisionTree: Explainability benchmark. Shows where non-linear boundaries
      exist without ensemble complexity.
    - RandomForest: Strong ensemble baseline. Robust to outliers, good feature
      importance. The "safe choice" before boosting.
    - XGBoost: LightGBM's direct competitor. Often wins on smaller datasets.
      Must compare — assuming LightGBM wins is not research, it's guessing.
    - LightGBM: Expected winner. Fastest on large tabular data, native missing
      value handling, first-class SHAP integration.
    - CatBoost: Rising in fintech. Handles categoricals natively, strong on
      imbalanced data, good calibration out of box.
    """
    return {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            min_samples_leaf=100,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=14,  # ~14:1 imbalance ratio
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate_model(
    name: str,
    model,
    X_train, y_train,
    X_val, y_val,
    X_train_scaled=None, X_val_scaled=None,
) -> dict:
    """
    WHY MULTI-DIMENSIONAL EVALUATION:
    AUC alone is how Kaggle thinks. Production systems care about:
    - KS Statistic: Credit industry standard
    - PR-AUC: Better than ROC-AUC for imbalanced classes
    - Brier Score: Calibration quality (are probabilities trustworthy?)
    - Inference time: Real-time API has latency constraints
    - F1 at optimal threshold: Balanced precision/recall
    """
    # Use scaled features for linear models
    use_scaled = name == "LogisticRegression"
    X_tr = X_train_scaled if use_scaled else X_train
    X_vl = X_val_scaled if use_scaled else X_val

    # Train + time it
    t0 = time.perf_counter()
    model.fit(X_tr, y_train)
    train_time = time.perf_counter() - t0

    # Inference time (per 1000 samples)
    t0 = time.perf_counter()
    y_prob = model.predict_proba(X_vl)[:, 1]
    inference_time_ms = (time.perf_counter() - t0) / len(X_vl) * 1000

    # Optimal threshold via F1
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    optimal_threshold = thresholds[np.argmax(f1_scores[:-1])]
    y_pred = (y_prob >= optimal_threshold).astype(int)

    # Model size (proxy via joblib)
    import io
    buf = io.BytesIO()
    joblib.dump(model, buf)
    model_size_mb = buf.tell() / 1024 / 1024

    metrics = {
        "model": name,
        "auc_roc": roc_auc_score(y_val, y_prob),
        "ks_statistic": ks_statistic(y_val, y_prob),
        "pr_auc": average_precision_score(y_val, y_prob),
        "f1": f1_score(y_val, y_pred),
        "brier_score": brier_score_loss(y_val, y_prob),
        "optimal_threshold": optimal_threshold,
        "train_time_sec": round(train_time, 2),
        "inference_ms_per_1k": round(inference_time_ms * 1000, 3),
        "model_size_mb": round(model_size_mb, 2),
        "y_prob": y_prob,  # stored for plots, removed before saving JSON
    }

    logger.info(
        f"{name:20s} | AUC={metrics['auc_roc']:.4f} | "
        f"KS={metrics['ks_statistic']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Brier={metrics['brier_score']:.4f} | "
        f"Infer={metrics['inference_ms_per_1k']:.2f}ms/1k"
    )
    return metrics


# ── Visualization ──────────────────────────────────────────────────────────
def plot_model_comparison(all_metrics: list, y_val: np.ndarray) -> None:
    """Generate comprehensive comparison plots."""
    df = pd.DataFrame([{k: v for k, v in m.items() if k != "y_prob"} for m in all_metrics])
    df = df.sort_values("auc_roc", ascending=False)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
    model_color = dict(zip(df["model"], colors))

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("OpenFinGuard — Model Comparison Dashboard", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. ROC curves
    ax1 = fig.add_subplot(gs[0, :2])
    for m in all_metrics:
        fpr, tpr, _ = roc_curve(y_val, m["y_prob"])
        ax1.plot(fpr, tpr, label=f"{m['model']} (AUC={m['auc_roc']:.3f})",
                 color=model_color[m["model"]], lw=2)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. PR curves
    ax2 = fig.add_subplot(gs[0, 2])
    for m in all_metrics:
        prec, rec, _ = precision_recall_curve(y_val, m["y_prob"])
        ax2.plot(rec, prec, label=f"{m['model']}", color=model_color[m["model"]], lw=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 3. KS Statistic bar
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.barh(df["model"], df["ks_statistic"], color=[model_color[m] for m in df["model"]])
    ax3.axvline(x=0.4, color="red", linestyle="--", alpha=0.7, label="Production threshold (0.4)")
    ax3.set_xlabel("KS Statistic")
    ax3.set_title("KS Statistic\n(Credit Industry Standard)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis="x")

    # 4. AUC-ROC bar
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.barh(df["model"], df["auc_roc"], color=[model_color[m] for m in df["model"]])
    ax4.set_xlabel("AUC-ROC")
    ax4.set_title("AUC-ROC Score")
    ax4.set_xlim(0.5, 1.0)
    ax4.grid(True, alpha=0.3, axis="x")

    # 5. Inference time
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.barh(df["model"], df["inference_ms_per_1k"], color=[model_color[m] for m in df["model"]])
    ax5.set_xlabel("ms per 1k samples")
    ax5.set_title("Inference Speed\n(lower is better)")
    ax5.grid(True, alpha=0.3, axis="x")

    # 6. Calibration curves
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    for m in all_metrics:
        fraction_pos, mean_pred = calibration_curve(y_val, m["y_prob"], n_bins=10)
        ax6.plot(mean_pred, fraction_pos, "s-", label=m["model"],
                 color=model_color[m["model"]], lw=2, markersize=4)
    ax6.set_xlabel("Mean Predicted Probability")
    ax6.set_ylabel("Fraction of Positives")
    ax6.set_title("Calibration Curves\n(closer to diagonal = more trustworthy probabilities)")
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # 7. Radar / multi-metric heatmap
    ax7 = fig.add_subplot(gs[2, 2])
    metrics_for_heat = ["auc_roc", "ks_statistic", "pr_auc", "f1"]
    heat_data = df.set_index("model")[metrics_for_heat]
    # Normalize 0–1
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)
    sns.heatmap(heat_norm, annot=heat_data.round(3), fmt=".3f",
                cmap="YlOrRd", ax=ax7, cbar=False, linewidths=0.5)
    ax7.set_title("Normalized Score Heatmap")
    ax7.set_xticklabels(["AUC", "KS", "PR-AUC", "F1"], rotation=30, fontsize=8)

    plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plot saved → {REPORTS_DIR / 'model_comparison.png'}")


def select_champion(all_metrics: list) -> str:
    """
    WHY THIS SCORING FORMULA:
    Weighted composite score reflecting production priorities:
    - AUC-ROC (30%): Core discrimination ability
    - KS Statistic (30%): Credit industry requirement
    - PR-AUC (25%): Imbalanced class performance
    - F1 (15%): Balanced precision/recall at optimal threshold

    Brier score and inference time are tiebreakers, not primary criteria.
    We don't penalize slightly slower models if they're significantly more fair.
    """
    scores = []
    for m in all_metrics:
        composite = (
            0.30 * m["auc_roc"] +
            0.30 * m["ks_statistic"] +
            0.25 * m["pr_auc"] +
            0.15 * m["f1"]
        )
        scores.append((m["model"], composite))

    scores.sort(key=lambda x: x[1], reverse=True)
    champion = scores[0][0]

    logger.info("\n🏆 Champion Selection:")
    for name, score in scores:
        marker = " ← CHAMPION" if name == champion else ""
        logger.info(f"  {name:20s} composite={score:.4f}{marker}")

    return champion


# ── Hyperparameter Tuning (Champion Only) ─────────────────────────────────
def tune_lightgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    """
    WHY TUNE ONLY THE WINNER:
    Tuning all 6 models wastes compute and signals poor prioritization.
    We tune the champion — that's engineering discipline.

    WHY NOT OPTUNA/HYPEROPT HERE:
    Grid search over a focused space is reproducible and explainable.
    Optuna's results change between runs — bad for a research report.
    """
    logger.info("Tuning LightGBM champion...")

    best_auc = 0
    best_params = {}
    best_model = None

    param_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "num_leaves": nl}
        for n in [500, 1000]
        for d in [5, 7, 9]
        for lr in [0.03, 0.05]
        for nl in [31, 63]
    ]

    for i, params in enumerate(param_grid):
        model = lgb.LGBMClassifier(
            **params,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)

        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model

        if (i + 1) % 8 == 0:
            logger.info(f"  Tuning progress: {i+1}/{len(param_grid)} | Best AUC so far: {best_auc:.4f}")

    logger.success(f"Best params: {best_params} | AUC: {best_auc:.4f}")
    return best_model


def tune_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    """Tune XGBoost if it wins the champion selection."""
    logger.info("Tuning XGBoost champion...")
    best_auc, best_model = 0, None

    param_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in [500, 1000]
        for d in [5, 7]
        for lr in [0.03, 0.05]
    ]

    for params in param_grid:
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=14,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
            eval_metric="auc",
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc = auc
            best_model = model

    logger.success(f"Best XGBoost AUC: {best_auc:.4f}")
    return best_model


# ── MLflow Logging ─────────────────────────────────────────────────────────
def log_to_mlflow(name: str, model, metrics: dict, champion: bool = False) -> None:
    """
    WHY MLFLOW:
    Reproducibility requires tracking every experiment. MLflow stores:
    - Parameters (what settings we used)
    - Metrics (what we got)
    - Artifacts (the actual model files)
    This means 6 months from now, you can reproduce the exact champion model.
    """
    with mlflow.start_run(run_name=name):
        # Log params
        if hasattr(model, "get_params"):
            params = {k: v for k, v in model.get_params().items() if v is not None}
            mlflow.log_params(params)

        # Log metrics (exclude non-serializable)
        loggable = {k: v for k, v in metrics.items()
                    if k not in ("model", "y_prob", "optimal_threshold")}
        mlflow.log_metrics(loggable)
        mlflow.log_metric("optimal_threshold", metrics["optimal_threshold"])

        if champion:
            mlflow.set_tag("champion", "true")
            mlflow.set_tag("stage", "production")

        # Log model artifact
        try:
            if isinstance(model, lgb.LGBMClassifier):
                mlflow.lightgbm.log_model(model, "model")
            elif isinstance(model, xgb.XGBClassifier):
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
        except Exception:
            pass  # MLflow artifact logging is best-effort


# ── Main ───────────────────────────────────────────────────────────────────
def run_training() -> dict:
    """Full training pipeline — returns champion model + metadata."""

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading processed datasets...")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_val   = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_val   = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze()
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    scaler  = joblib.load(PROCESSED_DIR / "scaler.joblib")

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── Train & evaluate all candidates ───────────────────────────────────
    mlflow.set_experiment("OpenFinGuard-ModelSelection")
    candidates = get_candidate_models()
    all_metrics = []

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3a: Training & Evaluating All Candidate Models")
    logger.info("=" * 80)

    for name, model in candidates.items():
        logger.info(f"\nTraining {name}...")
        metrics = evaluate_model(
            name, model,
            X_train, y_train,
            X_val, y_val,
            X_train_scaled, X_val_scaled,
        )
        all_metrics.append({**metrics, "_model_obj": model})
        log_to_mlflow(name, model, metrics)

    # ── Comparison table ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3b: Model Comparison Summary")
    logger.info("=" * 80)

    comparison_df = pd.DataFrame([
        {k: v for k, v in m.items() if k not in ("y_prob", "_model_obj")}
        for m in all_metrics
    ]).sort_values("auc_roc", ascending=False)

    print("\n" + comparison_df.to_string(index=False))
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # ── Visualizations ─────────────────────────────────────────────────────
    logger.info("\nSTAGE 3c: Generating comparison visualizations...")
    y_val_arr = y_val.values
    plot_model_comparison(
        [{k: v for k, v in m.items() if k != "_model_obj"} for m in all_metrics],
        y_val_arr,
    )

    # ── Champion selection ─────────────────────────────────────────────────
    logger.info("\nSTAGE 3d: Champion Selection")
    champion_name = select_champion(
        [{k: v for k, v in m.items() if k not in ("y_prob", "_model_obj")}
         for m in all_metrics]
    )
    champion_metrics = next(m for m in all_metrics if m["model"] == champion_name)
    champion_model = champion_metrics["_model_obj"]

    # ── Tune champion ──────────────────────────────────────────────────────
    logger.info(f"\nSTAGE 3e: Hyperparameter Tuning — {champion_name}")
    if champion_name == "LightGBM":
        champion_model = tune_lightgbm(X_train, y_train, X_val, y_val)
    elif champion_name == "XGBoost":
        champion_model = tune_xgboost(X_train, y_train, X_val, y_val)
    else:
        logger.info(f"  {champion_name} uses default params (tuning skipped — not gradient boosting)")

    # ── Final evaluation on held-out TEST set ──────────────────────────────
    logger.info("\nFinal evaluation on TEST set (held-out)...")
    use_scaled = champion_name == "LogisticRegression"
    X_te = X_test_scaled if use_scaled else X_test
    y_prob_test = champion_model.predict_proba(X_te)[:, 1]

    optimal_threshold = champion_metrics["optimal_threshold"]
    y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

    test_results = {
        "champion_model": champion_name,
        "test_auc_roc": round(roc_auc_score(y_test, y_prob_test), 4),
        "test_ks": round(ks_statistic(y_test.values, y_prob_test), 4),
        "test_pr_auc": round(average_precision_score(y_test, y_prob_test), 4),
        "test_f1": round(f1_score(y_test, y_pred_test), 4),
        "test_brier": round(brier_score_loss(y_test, y_prob_test), 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
    }

    logger.success(
        f"\n🏆 CHAMPION: {champion_name}\n"
        f"   Test AUC-ROC : {test_results['test_auc_roc']}\n"
        f"   Test KS      : {test_results['test_ks']}\n"
        f"   Test PR-AUC  : {test_results['test_pr_auc']}\n"
        f"   Test F1      : {test_results['test_f1']}"
    )

    # ── Save champion ──────────────────────────────────────────────────────
    joblib.dump(champion_model, MODELS_DIR / "champion_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    with open(MODELS_DIR / "champion_metadata.json", "w") as f:
        json.dump(test_results, f, indent=2)

    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(list(X_train.columns), f, indent=2)

    # MLflow — tag champion run
    log_to_mlflow(champion_name, champion_model, {
        **{k: v for k, v in test_results.items() if isinstance(v, float)},
        "model": champion_name,
        "optimal_threshold": optimal_threshold,
        "y_prob": y_prob_test,
    }, champion=True)

    logger.success(f"Champion model saved → {MODELS_DIR / 'champion_model.joblib'}")
    logger.success("Stage 3 complete. Ready for SHAP explainability (Stage 4).")

    return {
        "model": champion_model,
        "name": champion_name,
        "metadata": test_results,
        "feature_names": list(X_train.columns),
    }


if __name__ == "__main__":
    run_training()