import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# 1. LOAD & INSPECT
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(r'data\fpl_training_dataset.csv')

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nTarget distribution:\n", df["total_points"].describe())

# ─────────────────────────────────────────────────────────────
# 2. FEATURES & TARGET
# ─────────────────────────────────────────────────────────────
TARGET = "total_points"
DROP   = ["player_id", "total_points"]      # IDs / target — not features

X = df.drop(columns=DROP)
y = df[TARGET]

# was_home is bool → cast to int
X["was_home"] = X["was_home"].astype(int)

# Fill any NaNs with column median
X = X.fillna(X.median(numeric_only=True))

print(f"\nFeatures ({X.shape[1]}): {X.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / VALIDATION / TEST SPLIT  (60 / 20 / 20)
# ─────────────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"\nSplit — Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# 4. EVALUATION HELPER
# ─────────────────────────────────────────────────────────────
results = {}

def evaluate(name, model, X_ev, y_ev, split="Val"):
    preds = model.predict(X_ev)
    mae  = mean_absolute_error(y_ev, preds)
    rmse = mean_squared_error(y_ev, preds) ** 0.5
    r2   = r2_score(y_ev, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"\n[{name}] {split}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return preds

# ─────────────────────────────────────────────────────────────
# 5. RANDOM FOREST — baseline
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RANDOM FOREST — baseline")
print("="*55)

rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
evaluate("RF_baseline", rf_base, X_val, y_val)

# ─────────────────────────────────────────────────────────────
# 6. XGBOOST — baseline
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("XGBOOST — baseline")
print("="*55)

xgb_base = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_base.fit(X_train, y_train)
evaluate("XGB_baseline", xgb_base, X_val, y_val)

# ─────────────────────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING — GridSearchCV
# ─────────────────────────────────────────────────────────────

# --- 7a. Random Forest ---
print("\n" + "="*55)
print("TUNING — Random Forest (GridSearchCV)")
print("="*55)

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features":      ["sqrt", 0.5],
    },
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1,
)
rf_grid.fit(X_train, y_train)
print("Best RF params:", rf_grid.best_params_)
rf_tuned = rf_grid.best_estimator_
evaluate("RF_tuned", rf_tuned, X_val, y_val)

# --- 7b. XGBoost ---
print("\n" + "="*55)
print("TUNING — XGBoost (GridSearchCV)")
print("="*55)

xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    {
        "n_estimators":     [200, 400],
        "max_depth":        [4, 6],
        "learning_rate":    [0.03, 0.05],
        "subsample":        [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    },
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1,
)
xgb_grid.fit(X_train, y_train)
print("Best XGB params:", xgb_grid.best_params_)
xgb_tuned = xgb_grid.best_estimator_
evaluate("XGB_tuned", xgb_tuned, X_val, y_val)

# ─────────────────────────────────────────────────────────────
# 8. ENSEMBLES
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ENSEMBLES")
print("="*55)

# 8a. Voting — simple average of both tuned models
voting = VotingRegressor(estimators=[("rf", rf_tuned), ("xgb", xgb_tuned)])
voting.fit(X_train, y_train)
evaluate("Ensemble_Voting", voting, X_val, y_val)

# 8b. Stacking — Ridge meta-learner on top
stacking = StackingRegressor(
    estimators=[("rf", rf_tuned), ("xgb", xgb_tuned)],
    final_estimator=Ridge(alpha=1.0),
    cv=3,
    n_jobs=-1,
)
stacking.fit(X_train, y_train)
evaluate("Ensemble_Stacking", stacking, X_val, y_val)

# ─────────────────────────────────────────────────────────────
# 9. FINAL TEST EVALUATION — best model only
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL TEST SET EVALUATION")
print("="*55)

best_name = min(results, key=lambda k: results[k]["MAE"])
print(f"Best model by Val MAE: {best_name}")

model_map = {
    "RF_baseline":       rf_base,
    "XGB_baseline":      xgb_base,
    "RF_tuned":          rf_tuned,
    "XGB_tuned":         xgb_tuned,
    "Ensemble_Voting":   voting,
    "Ensemble_Stacking": stacking,
}
test_preds = evaluate(best_name, model_map[best_name], X_test, y_test, split="TEST")

# ─────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────

# 10a. Model comparison
results_df = pd.DataFrame(results).T.sort_values("MAE")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results_df))]
    results_df[metric].plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Model Comparison — Validation Set", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(r'data\model_comparison.png', bbox_inches="tight", dpi=150)
plt.show()

# 10b. Predicted vs Actual
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, test_preds, alpha=0.3, s=15, color="#3498db")
lims = [min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Points")
ax.set_ylabel("Predicted Points")
ax.set_title(f"{best_name} — Predicted vs Actual (Test Set)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(r'data\pred_vs_actual.png', bbox_inches="tight", dpi=150)
plt.show()

# 10c. Feature importances
importances = pd.Series(
    xgb_tuned.feature_importances_, index=X.columns
).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", ax=ax, color="#e74c3c", edgecolor="white")
ax.invert_yaxis()
ax.set_title("Top 15 Feature Importances (XGBoost Tuned)", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(r'data\feature_importance.png', bbox_inches="tight", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────
# 11. SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FULL RESULTS SUMMARY (Validation Set)")
print("="*55)
print(results_df.round(4).to_string())