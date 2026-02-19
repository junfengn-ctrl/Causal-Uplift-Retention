#!/usr/bin/env python3
# coding: utf-8
"""
Uplift Modeling for Customer Retention (single-file)
- T-Learner vs X-Learner
- Full EDA, plotting (retain original ~11 figures), SHAP, Qini, AUUC, Oracle ranking
- Business profit curve
Author: Junfeng Nie
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import time

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from scipy.stats import spearmanr


# ============================================================
# Console Utilities
# ============================================================

LINE_WIDTH = 70


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * LINE_WIDTH)
    print(title)
    print("=" * LINE_WIDTH)


def print_subsection(title: str):
    """Print formatted subsection header."""
    print("\n" + "-" * LINE_WIDTH)
    print(title)
    print("-" * LINE_WIDTH)


def print_metric_block(title: str, metrics: dict):
    """Pretty print metric dictionary."""
    print("\n" + "-" * LINE_WIDTH)
    print(title)
    print("-" * LINE_WIDTH)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<30}: {v:.4f}")
        else:
            print(f"{k:<30}: {v}")


# ---------------------------
# Global settings
# ---------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.style.use("ggplot")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)

# matplotlib warning noise suppression
warnings.filterwarnings("ignore", category=UserWarning)
# shap sometimes triggers matplotlib messages; keep them quiet for cleaner output
warnings.filterwarnings("ignore", module="shap")

# ---------------------------
# 1. Load data & simulate
# ---------------------------


def load_and_simulate_data(file_path: str) -> pd.DataFrame:
    """
    Load Telco churn data and simulate:
      - Randomized treatment assignment (RCT)
      - True individual treatment effect (oracle CATE)
      - Binary churn outcome under treatment/control (Churn_Simulated)
    Keep logic similar to original implementation.
    """
    df = pd.read_csv(file_path)

    # Basic cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Randomized treatment assignment
    df["Treatment"] = np.random.binomial(1, 0.5, size=len(df)).astype(int)

    # True individual uplift (using apply to preserve per-row random noise behavior)
    def compute_true_uplift(row):
        uplift = (
            0.20 * (row["Contract"] == "Month-to-month")
            + 0.002 * (row["MonthlyCharges"] - 50)
            - 0.05 * (row["tenure"] > 24)
        )
        uplift += np.random.normal(0, 0.02)
        return np.clip(uplift, -0.4, 0.4)

    df["true_uplift"] = df.apply(compute_true_uplift, axis=1)

    # Outcome simulation (row-wise to match original randomness)
    def simulate_outcome(row):
        prob = (
            0.25
            + 0.25 * (row["Contract"] == "Month-to-month")
            - 0.015 * (row["tenure"] / 12)
            + 0.002 * row["MonthlyCharges"]
        )
        prob += np.random.normal(0, 0.06)

        if row["Treatment"] == 1:
            prob -= row["true_uplift"]

        prob = np.clip(prob, 0.01, 0.99)
        return np.random.binomial(1, prob)

    df["Churn_Simulated"] = df.apply(simulate_outcome, axis=1).astype(int)

    return df


# ---------------------------
# 2. EDA
# ---------------------------


def run_eda(df: pd.DataFrame) -> None:
    """Run descriptive EDA with prints and plots similar to original script."""

    print("\n" + "=" * 70)
    print("EDA: Dataset Preview")
    print("=" * 70)

    # a. Head (first 3 rows)
    print("\n[1] First 3 Rows of the Dataset:")
    print(df.head(3))

    # b. DataFrame info
    print("\n" + "=" * 70)
    print("[2] DataFrame Info")
    print("=" * 70)
    df.info()

    # c. Missing values
    print("\n" + "=" * 70)
    print("[3] Missing Values per Column")
    print("=" * 70)
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    # d. Churn distribution pie chart
    churn_counts = df["Churn"].value_counts().sort_index()
    plt.figure(figsize=(6, 6))
    plt.pie(churn_counts, labels=["No Churn", "Churn"], autopct="%1.1f%%", startangle=90, shadow=True)
    plt.title("Churn Distribution (Pie Chart)")
    plt.show()

    # e. Treatment assignment check
    treat_counts = df["Treatment"].value_counts(normalize=True)
    print("\nTreatment assignment (proportions):")
    print(treat_counts)

    plt.figure(figsize=(6, 4))
    treat_counts.plot(kind="bar")
    plt.title("Treatment Assignment Distribution")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.show()

    # f. Churn rate by contract type (bar with annotations)
    churn_by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(churn_by_contract.index, churn_by_contract.values, color="#f67280")
    plt.xticks(rotation=25, ha="right", fontsize=11)
    plt.ylabel("Churn Rate", fontsize=12)
    plt.title("Churn Rate by Contract Type", fontsize=14)
    plt.ylim(0, max(churn_by_contract.values) * 1.15)
    for bar, val in zip(bars, churn_by_contract.values):
        height = bar.get_height()
        plt.annotate(f"{val:.2%}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.show()

    # g. MonthlyCharges by churn (boxplot)
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges by Churn")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.show()

    # h. Tenure distribution by churn
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, stat="density", common_norm=False)
    plt.title("Tenure Distribution by Churn")
    plt.show()

    print("\n" + "=" * 70)
    print("EDA Completed")
    print("=" * 70)


# ---------------------------
# 3. Feature engineering & split
# ---------------------------


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepare features (one-hot encoding) and split into train/test sets.
    Returns: X_train, X_test, y_train, y_test, t_train, t_test, true_uplift_train, true_uplift_test
    """
    X = df.drop(
        columns=["customerID", "Churn", "Churn_Simulated", "Treatment", "true_uplift"]
    )
    X = pd.get_dummies(X, drop_first=False)

    y = df["Churn_Simulated"]
    t = df["Treatment"]
    true_uplift = df["true_uplift"]

    return train_test_split(
        X, y, t, true_uplift, test_size=0.3, random_state=RANDOM_STATE
    )


# ---------------------------
# 4. X-Learner class
# ---------------------------


class XLearner(BaseEstimator, RegressorMixin):
    """X-Learner implementation using base learners (regressors)."""

    def __init__(self, model_class=xgb.XGBRegressor, params=None):
        self.model_class = model_class
        self.params = params or {}
        self.model_0 = None
        self.model_1 = None
        self.model_tau0 = None
        self.model_tau1 = None
        self.propensity = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series):
        # Base outcome models
        self.model_0 = self.model_class(**self.params)
        self.model_1 = self.model_class(**self.params)

        self.model_0.fit(X[t == 0], y[t == 0])
        self.model_1.fit(X[t == 1], y[t == 1])

        p0_pred = self.model_0.predict(X)
        p1_pred = self.model_1.predict(X)

        # Pseudo outcomes
        D1 = p0_pred[t == 1] - y[t == 1]
        D0 = y[t == 0] - p1_pred[t == 0]

        self.model_tau1 = self.model_class(**self.params)
        self.model_tau0 = self.model_class(**self.params)

        self.model_tau1.fit(X[t == 1], D1)
        self.model_tau0.fit(X[t == 0], D0)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_tau0 is None or self.model_tau1 is None:
            raise Exception("Model not fitted yet!")
        tau0 = self.model_tau0.predict(X)
        tau1 = self.model_tau1.predict(X)
        return self.propensity * tau0 + (1 - self.propensity) * tau1


# ---------------------------
# 5. SHAP interpretability
# ---------------------------


def plot_shap_for_uplift(model: XLearner, X_sample: pd.DataFrame) -> None:
    """Plot SHAP summary for model.model_tau1 (treatment pseudo-outcome model)."""
    print("\n" + "=" * 50)
    print("Interpretability: SHAP Values for Uplift Drivers")
    print("=" * 50)

    # Ensure sample is DataFrame
    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample)

    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model.model_tau1)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    plt.title("Key Drivers of Uplift (SHAP Summary)")
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.show()


# ---------------------------
# 6. Qini and AUUC
# ---------------------------


def build_qini(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    """Construct Qini table similar to original implementation."""
    df_res = pd.DataFrame({"uplift": uplift, "treatment": treatment, "outcome": outcome})
    df_res = df_res.sort_values("uplift", ascending=False).reset_index(drop=True)

    df_res["cum_treat"] = (df_res["treatment"] == 1).cumsum()
    df_res["cum_ctrl"] = (df_res["treatment"] == 0).cumsum()

    df_res["cum_y_treat"] = (df_res["outcome"] * (df_res["treatment"] == 1)).cumsum()
    df_res["cum_y_ctrl"] = (df_res["outcome"] * (df_res["treatment"] == 0)).cumsum()

    # Avoid division by zero
    df_res["qini"] = df_res["cum_treat"] * (
        (df_res["cum_y_ctrl"] / df_res["cum_ctrl"].clip(lower=1)) -
        (df_res["cum_y_treat"] / df_res["cum_treat"].clip(lower=1))
    )

    return df_res.replace([np.inf, -np.inf], 0).fillna(0)


def qini_metrics(results: pd.DataFrame) -> Tuple[float, float]:
    """Return AUUC and normalized Qini coefficient."""
    N = len(results)
    x = np.arange(N)
    auuc_model = auc(x, results["qini"])
    random_qini = np.linspace(0, results["qini"].iloc[-1], N)
    auuc_random = auc(x, random_qini)
    qini_coeff = (auuc_model - auuc_random) / (abs(auuc_random) + 1e-12)
    return auuc_model, qini_coeff


# ---------------------------
# 7. Plotting helpers
# ---------------------------


def plot_qini_curves(results_t: pd.DataFrame, results_x: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(results_t["qini"], label="T-Learner", linewidth=2)
    plt.plot(results_x["qini"], label="X-Learner", linewidth=2)
    plt.plot(np.linspace(0, results_t["qini"].iloc[-1], len(results_t)), linestyle="--", label="Random")
    plt.title("Qini Curve Comparison")
    plt.xlabel("Number of Targeted Users")
    plt.ylabel("Incremental Retention")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_qini_bar(qini_t: float, qini_x: float) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(["T-Learner", "X-Learner"], [qini_t, qini_x])
    plt.axhline(0, linestyle="--")
    plt.title("Normalized Qini Coefficient")
    plt.ylabel("Qini Coefficient")
    plt.show()


def plot_cate_calibration(pred: np.ndarray, true: np.ndarray) -> None:
    cate_df = pd.DataFrame({"pred": pred, "true": true})
    # deciles 0..9 -> reverse so 1 is highest
    cate_df["decile"] = pd.qcut(cate_df["pred"], 10, labels=False, duplicates="drop")
    cate_df["decile"] = 10 - cate_df["decile"].astype(int)
    summary = cate_df.groupby("decile").mean().sort_index()

    plt.figure(figsize=(8, 5))
    plt.plot(summary.index, summary["true"], marker="o", label="True CATE")
    plt.plot(summary.index, summary["pred"], marker="s", label="Predicted CATE")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Uplift Decile (1 = Highest)")
    plt.ylabel("Uplift (Control − Treatment)")
    plt.title("CATE Calibration by Decile (X-Learner)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_topk_gain(results_x: pd.DataFrame) -> None:
    percentages = np.arange(0.05, 0.55, 0.05)
    gains = []
    N = len(results_x)
    for p in percentages:
        k = int(p * N) - 1
        k = max(0, min(k, N - 1))
        gains.append(results_x["qini"].iloc[k])
    plt.figure(figsize=(8, 5))
    plt.plot(percentages * 100, gains, marker="o")
    plt.xlabel("Top % Users Targeted")
    plt.ylabel("Incremental Retention")
    plt.title("Business Gain vs Targeting Budget (X-Learner)")
    plt.grid(True)
    plt.show()


def plot_profit_curve(uplift_scores: np.ndarray, y_true: np.ndarray, t_true: np.ndarray,
                      value_per_retain: float = 60, cost_per_treat: float = 5) -> None:
    print("\n" + "=" * 50)
    print("Business Value: Profit Curve Analysis")
    print("=" * 50)

    df_res = pd.DataFrame({'uplift': uplift_scores, 'y': y_true, 't': t_true})
    df_res = df_res.sort_values('uplift', ascending=False).reset_index(drop=True)

    N = len(df_res)
    df_res['n_treat'] = df_res['t'].cumsum()
    df_res['n_ctrl'] = (1 - df_res['t']).cumsum()

    df_res['churn_treat'] = (df_res['y'] * df_res['t']).cumsum()
    df_res['churn_ctrl'] = (df_res['y'] * (1 - df_res['t'])).cumsum()

    r_treat = df_res['churn_treat'] / df_res['n_treat'].clip(lower=1)
    r_ctrl = df_res['churn_ctrl'] / df_res['n_ctrl'].clip(lower=1)

    n_targeted = np.arange(1, N + 1)
    current_uplift = r_ctrl - r_treat
    expected_gain = (current_uplift * n_targeted * value_per_retain)
    total_cost = n_targeted * cost_per_treat
    profit = expected_gain - total_cost

    max_profit_idx = int(np.nanargmax(profit))
    max_profit = float(profit.iloc[max_profit_idx])
    optimal_users = max_profit_idx + 1
    optimal_pct = optimal_users / N

    plt.figure(figsize=(10, 6))
    plt.plot(n_targeted, profit, label='Estimated Profit', linewidth=2)
    plt.axvline(optimal_users, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_pct:.1%}')
    plt.scatter(optimal_users, max_profit, color='red', s=100, zorder=5)
    plt.title(f"Profit Curve (Max Profit = ${max_profit:,.0f})")
    plt.xlabel("Number of Users Targeted")
    plt.ylabel("Expected Profit ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Strategy: Target top {optimal_pct:.1%} users.")
    print(f"Expected Max Profit: ${max_profit:,.2f}")


# ---------------------------
# 8. Main pipeline
# ---------------------------


def main():
    start_time = time.time()

    # ------------------------------------------------------------
    # Stage 1: Load & Simulate Data
    # ------------------------------------------------------------
    print_section("[Stage 1] Load & Simulate Data")

    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = load_and_simulate_data(data_path)
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    # ------------------------------------------------------------
    # Stage 2: Exploratory Data Analysis
    # ------------------------------------------------------------
    print_section("[Stage 2] Exploratory Data Analysis")
    run_eda(df)

    # ------------------------------------------------------------
    # Stage 3: Feature Engineering & Split
    # ------------------------------------------------------------
    print_section("[Stage 3] Feature Engineering & Train/Test Split")

    X_train, X_test, y_train, y_test, t_train, t_test, \
        true_uplift_train, true_uplift_test = prepare_features(df)

    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size : {X_test.shape[0]}")

    # ------------------------------------------------------------
    # Stage 4: T-Learner Training
    # ------------------------------------------------------------
    print_section("[Stage 4] T-Learner Training")

    params_clf = {
        "objective": "binary:logistic",
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    model_control = xgb.XGBClassifier(**params_clf)
    model_treat = xgb.XGBClassifier(**params_clf)

    model_control.fit(X_train[t_train == 0], y_train[t_train == 0])
    model_treat.fit(X_train[t_train == 1], y_train[t_train == 1])

    p0_test = model_control.predict_proba(X_test)[:, 1]
    p1_test = model_treat.predict_proba(X_test)[:, 1]
    uplift_t = p0_test - p1_test

    print("T-Learner training complete.")

    # ------------------------------------------------------------
    # Stage 5: X-Learner Training
    # ------------------------------------------------------------
    print_section("[Stage 5] X-Learner Training")

    params_xgb_reg = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }

    xl = XLearner(model_class=xgb.XGBRegressor, params=params_xgb_reg)
    xl.fit(X_train, y_train, t_train)
    uplift_x = xl.predict(X_test)

    print("X-Learner training complete.")

    # ------------------------------------------------------------
    # Stage 6: SHAP Interpretation
    # ------------------------------------------------------------
    print_section("[Stage 6] Model Interpretation (SHAP)")

    sample_size = min(500, len(X_test))
    plot_shap_for_uplift(
        xl,
        X_test.sample(sample_size, random_state=RANDOM_STATE)
    )

    # ------------------------------------------------------------
    # Stage 7: Uplift Evaluation
    # ------------------------------------------------------------
    print_section("[Stage 7] Uplift Evaluation")

    results_t = build_qini(uplift_t, t_test.values, y_test.values)
    results_x = build_qini(uplift_x, t_test.values, y_test.values)

    auuc_t, qini_t = qini_metrics(results_t)
    auuc_x, qini_x = qini_metrics(results_x)

    print_metric_block(
        "Model Comparison",
        {
            "T-Learner AUUC": auuc_t,
            "X-Learner AUUC": auuc_x,
            "T-Learner Qini": qini_t,
            "X-Learner Qini": qini_x,
        }
    )

    # Oracle ranking
    s_t, k_t = spearmanr(uplift_t, true_uplift_test)
    s_x, k_x = spearmanr(uplift_x, true_uplift_test)

    print_metric_block(
        "Oracle Ranking",
        {
            "T Spearman": s_t,
            "T Kendall": k_t,
            "X Spearman": s_x,
            "X Kendall": k_x,
        }
    )

    # ------------------------------------------------------------
    # Stage 8: Visualization
    # ------------------------------------------------------------
    print_section("[Stage 8] Visualization")

    plot_qini_curves(results_t, results_x)
    plot_qini_bar(qini_t, qini_x)
    plot_cate_calibration(uplift_x, true_uplift_test.values)
    plot_topk_gain(results_x)

    # ------------------------------------------------------------
    # Stage 9: Business Value Analysis
    # ------------------------------------------------------------
    print_section("[Stage 9] Business Value Analysis")

    plot_profit_curve(
        uplift_x,
        y_test.values,
        t_test.values,
        value_per_retain=60,
        cost_per_treat=5,
    )

    # ------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------
    total_time = time.time() - start_time
    print_section("Pipeline Completed Successfully")
    print(f"Total runtime: {total_time:.2f} seconds")



if __name__ == "__main__":
    main()
