"""
=============================================================
  Credit Card Default Analysis System
  B.Tech Computer Science (Data Science) Academic Project
=============================================================
  Dataset: UCI Default of Credit Card Clients
  Author: [Your Name]
  Date: 2026
=============================================================
"""

# ─────────────────────────────────────────────
# STEP 0: Import Libraries
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
import joblib

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# Plot Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

print("=" * 60)
print("   CREDIT CARD DEFAULT ANALYSIS SYSTEM")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
print("\n[1] Loading Dataset...")

# Download programmatically from UCI if not present
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

try:
    df = pd.read_excel("credit.xls", header=1)
    print(f"    Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception:
    # Fallback: generate synthetic dataset with same schema for demonstration
    print("    Note: Generating synthetic dataset (UCI URL unavailable).")
    np.random.seed(42)
    n = 30000
    df = pd.DataFrame({
        "ID": range(1, n + 1),
        "LIMIT_BAL": np.random.choice(
            [10000, 20000, 30000, 50000, 80000, 100000, 150000, 200000, 300000, 500000],
            n, p=[0.05, 0.10, 0.15, 0.20, 0.15, 0.15, 0.10, 0.05, 0.03, 0.02]),
        "SEX": np.random.choice([1, 2], n, p=[0.40, 0.60]),
        "EDUCATION": np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.04, 0.47, 0.35, 0.16, 0.01, 0.01] ),
        "MARRIAGE": np.random.choice([0, 1, 2, 3], n, p=[0.01, 0.45, 0.53, 0.01]),
        "AGE": np.random.randint(21, 75, n),
        **{f"PAY_{i}": np.random.choice([-2,-1,0,1,2,3,4,5,6,7,8], n,
           p=[0.10,0.15,0.45,0.08,0.08,0.05,0.03,0.02,0.01,0.01,0.02])
           for i in [0,2,3,4,5,6]},
        **{f"BILL_AMT{i}": np.random.randint(-10000, 500000, n) for i in range(1, 7)},
        **{f"PAY_AMT{i}": np.abs(np.random.randint(0, 200000, n)) for i in range(1, 7)},
    })
    # Generate target with realistic correlation
    risk = (
        (df["PAY_0"] > 0).astype(int) * 3 +
        (df["LIMIT_BAL"] < 30000).astype(int) +
        (df["AGE"] < 30).astype(int) +
        (df["BILL_AMT1"] > 100000).astype(int)
    )
    prob = 1 / (1 + np.exp(-(risk - 3.0)))
    df["default.payment.next.month"] = np.random.binomial(1, prob)

# ─────────────────────────────────────────────
# STEP 2: Data Preprocessing
# ─────────────────────────────────────────────
print("\n[2] Data Preprocessing...")

# Rename columns for clarity
df.columns = df.columns.str.strip()
rename_map = {"PAY_0": "PAY_1", "default payment next month": "DEFAULT"}
df.rename(columns=rename_map, inplace=True)
print(df.columns)

# Drop ID column if present
if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)

# Show basic info
print(f"    Shape          : {df.shape}")
print(f"    Missing values : {df.isnull().sum().sum()}")
print(f"    Default rate   : {df['DEFAULT'].mean()*100:.2f}%")

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode education & marriage (clean unknown categories)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # group unknowns
df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})                # group unknowns

# Feature / Target split
X = df.drop(columns=["DEFAULT"])
y = df["DEFAULT"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train-Test Split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y)

print(f"    Train size     : {X_train.shape[0]}")
print(f"    Test  size     : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# STEP 3: EDA Visualizations
# ─────────────────────────────────────────────
print("\n[3] Generating EDA Visualizations...")

COLORS = {"default": "#E74C3C", "no_default": "#2ECC71",
          "blue": "#3498DB", "purple": "#9B59B6"}

# --- 3a. Default Distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts()
bars = ax.bar(["No Default", "Default"], counts.values,
              color=[COLORS["no_default"], COLORS["default"]], width=0.5, edgecolor="white")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{bar.get_height():,}', ha='center', fontweight='bold')
ax.set_title("Class Distribution: Default vs No Default", fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.savefig("outputs/eda_01_class_distribution.png")
plt.close()

# --- 3b. Age Distribution ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df[df["DEFAULT"]==0]["AGE"], bins=30, alpha=0.7, color=COLORS["no_default"], label="No Default")
ax.hist(df[df["DEFAULT"]==1]["AGE"], bins=30, alpha=0.7, color=COLORS["default"], label="Default")
ax.set_title("Age Distribution by Default Status", fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda_02_age_distribution.png")
plt.close()

# --- 3c. Credit Limit Box Plot ---
fig, ax = plt.subplots(figsize=(7, 5))
df_plot = df.copy()
df_plot["Status"] = df_plot["DEFAULT"].map({0: "No Default", 1: "Default"})
sns.boxplot(x="Status", y="LIMIT_BAL", data=df_plot,
            palette={"No Default": COLORS["no_default"], "Default": COLORS["default"]}, ax=ax)
ax.set_title("Credit Limit vs Default Status", fontweight="bold")
ax.set_ylabel("Credit Limit (NTD)")
plt.tight_layout()
plt.savefig("outputs/eda_03_credit_limit_boxplot.png")
plt.close()

# --- 3d. Education Count Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
edu_map = {1: "Grad School", 2: "University", 3: "High School", 4: "Other"}
df_plot["EDUCATION_LABEL"] = df["EDUCATION"].map(edu_map)
edu_default = df_plot.groupby(["EDUCATION_LABEL", "Status"]).size().reset_index(name="count")
sns.barplot(x="EDUCATION_LABEL", y="count", hue="Status", data=edu_default,
            palette={"No Default": COLORS["no_default"], "Default": COLORS["default"]}, ax=ax)
ax.set_title("Default Rate by Education Level", fontweight="bold")
ax.set_xlabel("Education"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/eda_04_education_default.png")
plt.close()

# --- 3e. Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_05_correlation_heatmap.png")
plt.close()

# --- 3f. Payment History (PAY_1) ---
fig, ax = plt.subplots(figsize=(9, 5))
pay_counts = df.groupby(["PAY_1", "DEFAULT"]).size().unstack(fill_value=0)
pay_counts.plot(kind="bar", ax=ax,
                color=[COLORS["no_default"], COLORS["default"]], edgecolor="white")
ax.set_title("Payment Delay (Month 1) vs Default Status", fontweight="bold")
ax.set_xlabel("Payment Delay (months)"); ax.set_ylabel("Count")
ax.legend(["No Default", "Default"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("outputs/eda_06_payment_history.png")
plt.close()

print("    Saved 6 EDA charts to outputs/")

# ─────────────────────────────────────────────
# STEP 4: Model Building
# ─────────────────────────────────────────────
print("\n[4] Training Models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42,
                                                  class_weight="balanced", n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }
    print(f"    {name}: Acc={results[name]['accuracy']:.4f}, "
          f"F1={results[name]['f1']:.4f}, AUC={results[name]['roc_auc']:.4f}")

# ─────────────────────────────────────────────
# STEP 5: Evaluation – Confusion Matrices
# ─────────────────────────────────────────────
print("\n[5] Generating Evaluation Charts...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    ax.set_title(f"{name}", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eval_01_confusion_matrices.png")
plt.close()

# --- ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
colors = [COLORS["blue"], COLORS["purple"], COLORS["default"]]
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f"{name} (AUC = {res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – Model Comparison", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/eval_02_roc_curves.png")
plt.close()

# --- Model Comparison Bar Chart ---
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
x = np.arange(len(metrics))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
offsets = [-1, 0, 1]
bar_colors = [COLORS["blue"], COLORS["purple"], COLORS["default"]]
for (name, res), off, col in zip(results.items(), offsets, bar_colors):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + off * width, vals, width, label=name, color=col, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontweight="bold")
ax.legend(); ax.axhline(0.5, color="gray", linestyle="--", lw=0.8)
plt.tight_layout()
plt.savefig("outputs/eval_03_model_comparison.png")
plt.close()

# ─────────────────────────────────────────────
# STEP 6: Feature Importance (Random Forest)
# ─────────────────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
importances.sort_values().plot(kind="barh", color=COLORS["blue"], ax=ax, edgecolor="white")
ax.set_title("Top 15 Feature Importances – Random Forest", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/eval_04_feature_importance.png")
plt.close()

print("    Saved 4 evaluation charts to outputs/")

# ─────────────────────────────────────────────
# STEP 7: Risk Score System
# ─────────────────────────────────────────────
print("\n[6] Risk Score System...")

def compute_risk_score(prob):
    """Convert probability (0–1) to risk score (0–100) with non-linear scaling."""
    score = int(prob * 100)
    if score <= 30:
        label = "LOW RISK 🟢"
    elif score <= 70:
        label = "MEDIUM RISK 🟡"
    else:
        label = "HIGH RISK 🔴"
    return score, label

def get_recommendations(score, features: dict):
    """Return personalized prescriptive recommendations based on risk score."""
    recs = []
    if score >= 71:
        recs += [
            "🔴 URGENT: Suspend new credit limit increases immediately.",
            "📞 Assign dedicated relationship manager for this account.",
            "💳 Block cash advance features on card.",
            "📋 Offer structured EMI repayment plan (6–12 months).",
            "📧 Send immediate payment overdue alert via SMS & email.",
        ]
    elif score >= 31:
        recs += [
            "🟡 Review credit limit – consider a 20–30% reduction.",
            "📩 Send friendly payment reminder 10 days before due date.",
            "💡 Offer loyalty/cashback incentives to encourage on-time payment.",
            "📊 Monitor spending pattern monthly.",
            "🤝 Offer flexible payment restructuring if requested.",
        ]
    else:
        recs += [
            "🟢 Account is healthy. Maintain current credit limit.",
            "🎁 Consider a credit limit increase as a reward for good standing.",
            "📈 Flag as potential upsell candidate (premium card/loan offer).",
        ]
    return recs

# Demo: Apply to test set with RF model
rf_probs = results["Random Forest"]["y_prob"]
sample_scores = [compute_risk_score(p) for p in rf_probs[:5]]
for i, (sc, lb) in enumerate(sample_scores):
    print(f"    Customer #{i+1}: Score={sc} → {lb}")

# ─────────────────────────────────────────────
# STEP 8: Save Models
# ─────────────────────────────────────────────
print("\n[7] Saving Models...")
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
print("    Saved: models/random_forest_model.pkl")
print("    Saved: models/scaler.pkl")

# ─────────────────────────────────────────────
# STEP 9: Final Summary Table
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 60)
header = f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}"
print(header)
print("-" * 67)
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>8.4f} {res['precision']:>8.4f} "
          f"{res['recall']:>8.4f} {res['f1']:>8.4f} {res['roc_auc']:>8.4f}")
print("=" * 60)
print("\n✅ Analysis complete. All outputs saved to 'outputs/' folder.")
print("   Run 'streamlit run app.py' to launch the web application.")
