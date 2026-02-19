# ============================================================
# UPLIFT MODELING PROJECT
# T-Learner vs X-Learner with Full Evaluation & Visualization
# ============================================================

# ---------------------------
# 0. Libraries & Settings
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from scipy.stats import spearmanr, kendalltau

import xgboost as xgb

# Visualization style
plt.style.use("ggplot")
sns.set_palette("husl")

pd.set_option("display.max_columns", None)
np.random.seed(42)


# ---------------------------
# 1. Load Data & Causal Simulation
# ---------------------------
def load_and_simulate_data(file_path):
    """
    Load Telco churn data and simulate:
    - Randomized treatment (RCT)
    - Ground-truth heterogeneous treatment effects (true CATE)
    - Binary churn outcome under treatment/control
    """

    df = pd.read_csv(file_path)

    # Basic cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Randomized treatment assignment (RCT assumption)
    df["Treatment"] = np.random.binomial(1, 0.5, size=len(df))

    # True individual treatment effect (oracle CATE)
    def compute_true_uplift(row):
        uplift = (
            0.20 * (row["Contract"] == "Month-to-month")
            + 0.002 * (row["MonthlyCharges"] - 50)
            - 0.05 * (row["tenure"] > 24)
        )
        uplift += np.random.normal(0, 0.02)
        return np.clip(uplift, -0.4, 0.4)

    df["true_uplift"] = df.apply(compute_true_uplift, axis=1)

    # Outcome simulation: churn probability
    def simulate_outcome(row):
        prob = (
            0.25
            + 0.25 * (row["Contract"] == "Month-to-month")
            - 0.015 * (row["tenure"] / 12)
            + 0.002 * row["MonthlyCharges"]
        )
        prob += np.random.normal(0, 0.06)

        # Treatment reduces churn by true uplift
        if row["Treatment"] == 1:
            prob -= row["true_uplift"]

        prob = np.clip(prob, 0.01, 0.99)
        return np.random.binomial(1, prob)

    df["Churn_Simulated"] = df.apply(simulate_outcome, axis=1)

    return df


df = load_and_simulate_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ============================================================
# EDA: Exploratory Data Analysis
# Purpose:
# - Understand churn distribution
# - Check key covariates related to churn & uplift
# - Provide context before causal modeling
# ============================================================

def run_eda(df):
    """
    Run basic but meaningful EDA for churn & uplift analysis.
    """

    print("\n" + "=" * 70)
    print("EDA: Dataset Preview")
    print("=" * 70)

    # ---------------------------
    # a. Head (first 3 rows)
    # ---------------------------
    print("\n[1] First 3 Rows of the Dataset:")
    print(df.head(3))

    # ---------------------------
    # b. Data Structure & Types
    # ---------------------------
    print("\n" + "=" * 70)
    print("[2] DataFrame Info")
    print("=" * 70)
    df.info()

    # ---------------------------
    # c. Missing Values
    # ---------------------------
    print("\n" + "=" * 70)
    print("[3] Missing Values per Column")
    print("=" * 70)
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    # ---------------------------Yw
    # d. Churn Distribution (Pie Chart)
    # ---------------------------
    churn_counts = df["Churn"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(
        churn_counts,
        labels=["No Churn", "Churn"],
        autopct="%1.1f%%",
        startangle=90,
        shadow=True
    )
    plt.title("Churn Distribution (Pie Chart)")
    plt.show()

    # ---------------------------
    # e. Treatment Assignment Check
    # ---------------------------
    treat_counts = df["Treatment"].value_counts(normalize=True)
    print(treat_counts)

    plt.figure(figsize=(6, 4))
    treat_counts.plot(kind="bar")
    plt.title("Treatment Assignment Distribution")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.show()

    # ---------------------------
    # f. Churn Rate by Key Categorical Features
    # ---------------------------
    # --- Matplotlib 版本：放大 + 调整底部边距 + 在柱上标注数值 ---
    churn_by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))  # 放大画布
    bars = plt.bar(churn_by_contract.index, churn_by_contract.values, color='#f67280')

    # 让 x 标签倾斜并右对齐，防止被切掉
    plt.xticks(rotation=25, ha='right', fontsize=11)
    plt.ylabel("Churn Rate", fontsize=12)
    plt.title("Churn Rate by Contract Type", fontsize=14)

    # 设置 y 轴上限让柱子与标题/轴标签有缓冲
    plt.ylim(0, max(churn_by_contract.values) * 1.15)

    # 在每个柱子上添加数值标签（保留两位小数）
    for bar, val in zip(bars, churn_by_contract.values):
        height = bar.get_height()
        plt.annotate(f"{val:.2%}",  # 百分比格式
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),  # 标签向上偏移一点
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=11)

    plt.tight_layout()  # 自动调整边距，通常能避免被切
    # plt.savefig("churn_by_contract.png", bbox_inches="tight", dpi=150)  # 可选：保存图像
    plt.show()

    # ---------------------------
    # g. Numerical Feature vs Churn
    # ---------------------------

    plt.figure(figsize=(7, 4))
    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        data=df
    )
    plt.title("Monthly Charges by Churn")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.show()

    # ---------------------------
    # h. Tenure Distribution by Churn
    # ---------------------------

    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        bins=30,
        kde=True,
        stat="density",
        common_norm=False
    )
    plt.title("Tenure Distribution by Churn")
    plt.show()

    print("\n" + "=" * 70)
    print("EDA Completed")
    print("=" * 70)


run_eda(df)

# ---------------------------
# 2. Feature Engineering & Split
# ---------------------------
X = df.drop(columns=[
    "customerID",
    "Churn",
    "Churn_Simulated",
    "Treatment",
    "true_uplift"
])

X = pd.get_dummies(X, drop_first=False)

y = df["Churn_Simulated"]
t = df["Treatment"]
true_uplift = df["true_uplift"]

X_train, X_test, y_train, y_test, t_train, t_test, true_uplift_train, true_uplift_test = (
    train_test_split(
        X, y, t, true_uplift,
        test_size=0.3,
        random_state=42
    )
)


# ============================================================
# NEW SECTION: Custom X-Learner Implementation
# ============================================================
class XLearner(BaseEstimator, RegressorMixin):
    def __init__(self, model_class=xgb.XGBRegressor, params=None):
        self.model_class = model_class
        self.params = params if params else {}
        self.model_0 = None
        self.model_1 = None
        self.model_tau0 = None
        self.model_tau1 = None
        self.propensity = 0.5

    def fit(self, X, y, t):
        # Stage 1: Base Outcome Models
        self.model_0 = self.model_class(**self.params)
        self.model_1 = self.model_class(**self.params)

        self.model_0.fit(X[t == 0], y[t == 0])
        self.model_1.fit(X[t == 1], y[t == 1])

        p0_pred = self.model_0.predict(X)
        p1_pred = self.model_1.predict(X)

        # Stage 2: Pseudo-Outcome (Uplift)
        # Definition: Uplift = P(Churn|Control) - P(Churn|Treat)
        # Logic: If Treat reduces churn, p0 > p1, so Uplift > 0.
        D1 = p0_pred[t == 1] - y[t == 1]
        D0 = y[t == 0] - p1_pred[t == 0]

        self.model_tau1 = self.model_class(**self.params)
        self.model_tau0 = self.model_class(**self.params)

        self.model_tau1.fit(X[t == 1], D1)
        self.model_tau0.fit(X[t == 0], D0)

        return self

    def predict(self, X):
        if self.model_tau0 is None:
            raise Exception("Model not fitted yet!")
        tau0 = self.model_tau0.predict(X)
        tau1 = self.model_tau1.predict(X)
        # Propensity weighted average
        return self.propensity * tau0 + (1 - self.propensity) * tau1

# ---------------------------
# 3. T-Learner (Baseline)
# ---------------------------
params = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": 42
}

model_control = xgb.XGBClassifier(**params)
model_treat = xgb.XGBClassifier(**params)

model_control.fit(X_train[t_train == 0], y_train[t_train == 0])
model_treat.fit(X_train[t_train == 1], y_train[t_train == 1])

# T-learner CATE prediction
p0_test = model_control.predict_proba(X_test)[:, 1]
p1_test = model_treat.predict_proba(X_test)[:, 1]
uplift_t = p0_test - p1_test

# ---------------------------
# 4. X-Learner (Class Implementation) & Interpretation
# ---------------------------

# A. 定义参数 (沿用之前的 XGBoost 参数，改为回归任务)
params_x = {
    "objective": "reg:squarederror",  # 注意：第二阶段是回归任务
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": 42
}

# B. 训练 X-Learner (使用我们刚才定义的类)
print("Training X-Learner (Class-based)...")
# 初始化模型
xl = XLearner(model_class=xgb.XGBRegressor, params=params_x)
# 训练
xl.fit(X_train, y_train, t_train)
# 预测 Uplift Score (这一步替换了原来手算的 uplift_x)
uplift_x = xl.predict(X_test)


# C. SHAP 可解释性分析 (New Feature for DS)
def plot_shap_for_uplift(model, X_sample):
    print("\n" + "=" * 50)
    print("Interpretability: SHAP Values for Uplift Drivers")
    print("=" * 50)

    # 我们解释 model_tau1 (处理组的CATE模型)
    # 这张图回答：在 Treatment 组中，哪些特征最能驱动 Uplift？
    explainer = shap.TreeExplainer(model.model_tau1)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    plt.title("Key Drivers of Uplift (SHAP Summary)")
    # show=False 允许我们继续调整 plt 设置
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.show()


# 为了运行速度，我们只取 500 个样本做 SHAP 分析
# 注意：确保这里 X_test 是 DataFrame 格式
plot_shap_for_uplift(xl, X_test.sample(500, random_state=42))

print("X-Learner training and interpretation complete.")


# ---------------------------
# 5. Qini Curve Construction
# ---------------------------
def build_qini(uplift, treatment, outcome):
    df_res = pd.DataFrame({
        "uplift": uplift,
        "treatment": treatment,
        "outcome": outcome
    }).sort_values("uplift", ascending=False).reset_index(drop=True)

    df_res["cum_treat"] = (df_res["treatment"] == 1).cumsum()
    df_res["cum_ctrl"] = (df_res["treatment"] == 0).cumsum()

    df_res["cum_y_treat"] = (df_res["outcome"] * (df_res["treatment"] == 1)).cumsum()
    df_res["cum_y_ctrl"] = (df_res["outcome"] * (df_res["treatment"] == 0)).cumsum()

    df_res["qini"] = df_res["cum_treat"] * (
        (df_res["cum_y_ctrl"] / df_res["cum_ctrl"]) -
        (df_res["cum_y_treat"] / df_res["cum_treat"])
    )

    return df_res.replace([np.inf, -np.inf], 0).fillna(0)


results_t = build_qini(uplift_t, t_test.values, y_test.values)
results_x = build_qini(uplift_x, t_test.values, y_test.values)


# ---------------------------
# 6. AUUC & Qini Coefficient
# ---------------------------
def qini_metrics(results):
    N = len(results)
    x = np.arange(N)

    auuc_model = auc(x, results["qini"])
    random_qini = np.linspace(0, results["qini"].iloc[-1], N)
    auuc_random = auc(x, random_qini)

    qini_coeff = (auuc_model - auuc_random) / abs(auuc_random)
    return auuc_model, qini_coeff


auuc_t, qini_t = qini_metrics(results_t)
auuc_x, qini_x = qini_metrics(results_x)

print("=== Model Comparison ===")
print(f"T-Learner | AUUC: {auuc_t:.2f}, Qini Coef: {qini_t:.3f}")
print(f"X-Learner | AUUC: {auuc_x:.2f}, Qini Coef: {qini_x:.3f}")


# ---------------------------
# 7. Oracle CATE Ranking Validation
# ---------------------------
s_t, k_t = spearmanr(uplift_t, true_uplift_test), kendalltau(uplift_t, true_uplift_test)
s_x, k_x = spearmanr(uplift_x, true_uplift_test), kendalltau(uplift_x, true_uplift_test)

print("\n=== Oracle Ranking ===")
print(f"T-Learner | Spearman: {s_t[0]:.3f}, Kendall: {k_t[0]:.3f}")
print(f"X-Learner | Spearman: {s_x[0]:.3f}, Kendall: {k_x[0]:.3f}")


# ---------------------------
# 8. Visualization 1: Qini Curves
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(results_t["qini"], label="T-Learner", linewidth=2)
plt.plot(results_x["qini"], label="X-Learner", linewidth=2)
plt.plot(
    np.linspace(0, results_t["qini"].iloc[-1], len(results_t)),
    linestyle="--",
    label="Random"
)
plt.title("Qini Curve Comparison")
plt.xlabel("Number of Targeted Users")
plt.ylabel("Incremental Retention")
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------
# 9. Visualization 2: Qini Coefficient Bar Chart
# ---------------------------
plt.figure(figsize=(6, 4))
plt.bar(["T-Learner", "X-Learner"], [qini_t, qini_x])
plt.axhline(0, linestyle="--")
plt.title("Normalized Qini Coefficient")
plt.ylabel("Qini Coefficient")
plt.show()


# ---------------------------
# 10. Visualization 3: CATE Calibration (X-Learner)
# ---------------------------
cate_df = pd.DataFrame({
    "pred": uplift_x,
    "true": true_uplift_test.values
})

cate_df["decile"] = pd.qcut(cate_df["pred"], 10, labels=False)
cate_df["decile"] = 10 - cate_df["decile"]

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


# ---------------------------
# 11. Visualization 4: Top-K Targeting Gain
# ---------------------------
percentages = np.arange(0.05, 0.55, 0.05)
gains = []

for p in percentages:
    k = int(p * len(results_x))
    gains.append(results_x["qini"].iloc[k])

plt.figure(figsize=(8, 5))
plt.plot(percentages * 100, gains, marker="o")
plt.xlabel("Top % Users Targeted")
plt.ylabel("Incremental Retention")
plt.title("Business Gain vs Targeting Budget (X-Learner)")
plt.grid(True)
plt.show()


# ============================================================
# 12. Business Value: Profit Curve Analysis
# ============================================================
def plot_profit_curve(uplift_scores, y_true, t_true,
                      value_per_retain=60, cost_per_treat=5):
    print("\n" + "=" * 50)
    print("Business Value: Profit Curve Analysis")
    print("=" * 50)

    # DataFrame for analysis
    df_res = pd.DataFrame({
        'uplift': uplift_scores,
        'y': y_true,  # Actual Outcome (Churn=1)
        't': t_true  # Actual Treatment
    }).sort_values('uplift', ascending=False)

    N = len(df_res)

    # Calculate incremental gains
    # If we target top k users:
    # Cost = k * cost_per_treat
    # Gain = (Churn_Control - Churn_Treat) * value_per_retain
    # We estimate this using the cumulative difference in churn rates

    df_res['n_treat'] = df_res['t'].cumsum()
    df_res['n_ctrl'] = (1 - df_res['t']).cumsum()

    df_res['churn_treat'] = (df_res['y'] * df_res['t']).cumsum()
    df_res['churn_ctrl'] = (df_res['y'] * (1 - df_res['t'])).cumsum()

    # Cumulative conversion rates (prevent div by zero)
    r_treat = df_res['churn_treat'] / df_res['n_treat'].clip(lower=1)
    r_ctrl = df_res['churn_ctrl'] / df_res['n_ctrl'].clip(lower=1)

    # Estimated Lift * Population * Value - Cost
    n_targeted = np.arange(1, N + 1)
    current_uplift = r_ctrl - r_treat

    expected_gain = (current_uplift * n_targeted * value_per_retain)
    total_cost = n_targeted * cost_per_treat
    profit = expected_gain - total_cost

    # Plot
    max_profit_idx = np.argmax(profit)
    max_profit = profit.iloc[max_profit_idx]
    optimal_users = max_profit_idx + 1
    optimal_pct = optimal_users / N

    plt.figure(figsize=(10, 6))
    plt.plot(n_targeted, profit, label='Estimated Profit', color='green', linewidth=2)
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


# Run Profit Analysis
plot_profit_curve(uplift_x, y_test, t_test, value_per_retain=60, cost_per_treat=5)
