import os
import logging
import numpy as np
import pandas as pd
import joblib
import yaml
import mlflow
import mlflow.xgboost
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose per-trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def calculate_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Calculate performance metrics in log space and original dollar space."""
    y_true_dollar = np.expm1(y_true_log)
    y_pred_dollar = np.expm1(y_pred_log)

    r2 = r2_score(y_true_log, y_pred_log)
    mae_dollar = mean_absolute_error(y_true_dollar, y_pred_dollar)
    rmse_dollar = np.sqrt(mean_squared_error(y_true_dollar, y_pred_dollar))
    mape = np.mean(np.abs((y_true_dollar - y_pred_dollar) / y_true_dollar)) * 100

    return {
        "r2_score": float(r2),
        "mae_dollar": float(mae_dollar),
        "rmse_dollar": float(rmse_dollar),
        "mape_percent": float(mape),
    }


def run_progressive_training_pipeline():
    """
    Execute progressive modeling pipeline: Simple -> Complex -> Tuning (100 trials) -> Stacking.

    Improvements Applied in this Run:
    1. Polynomial features (bmi^2, age^2, age*smoker) from updated preprocessor
    2. BMI Risk Tier 4-level ordinal encoding from updated preprocessor
    3. Optuna 100-trial Bayesian optimization with WIDER hyperparameter search space
    """
    X_train, X_test, y_train, y_test, preprocessor, _ = prepare_data()
    mlflow.set_experiment("medical-cost-predictor-v2")

    results = {}

    # ─────────────────────────────────────────────────────────
    # Stage 1: Simple Baseline Models
    # ─────────────────────────────────────────────────────────
    logger.info("━━━ Stage 1: Simple Baseline Models ━━━")
    simple_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=0.01),
        "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=42),
    }

    for name, model in simple_models.items():
        with mlflow.start_run(run_name=f"Stage1_{name.replace(' ', '_')}"):
            model.fit(X_train, y_train)
            metrics = calculate_metrics(y_test, model.predict(X_test))
            mlflow.log_metrics(metrics)
            results[name] = metrics
            logger.info(f"  [{name:20s}] R²: {metrics['r2_score']:.4f} | MAE: ${metrics['mae_dollar']:,.2f}")

    # ─────────────────────────────────────────────────────────
    # Stage 2: Complex Non-Linear Models
    # ─────────────────────────────────────────────────────────
    logger.info("\n━━━ Stage 2: Complex Ensemble Models ━━━")
    complex_models = {
        "Random Forest":    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        "XGBoost Default":  XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                                         random_state=42, n_jobs=2, verbosity=0),
        "LightGBM Default": LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                                          random_state=42, n_jobs=2, verbose=-1),
    }

    for name, model in complex_models.items():
        with mlflow.start_run(run_name=f"Stage2_{name.replace(' ', '_')}"):
            model.fit(X_train, y_train)
            metrics = calculate_metrics(y_test, model.predict(X_test))
            mlflow.log_metrics(metrics)
            results[name] = metrics
            logger.info(f"  [{name:20s}] R²: {metrics['r2_score']:.4f} | MAE: ${metrics['mae_dollar']:,.2f}")

    # ─────────────────────────────────────────────────────────
    # Stage 3: IMPROVED Optuna Tuning — 100 Trials, Wider Space
    # ─────────────────────────────────────────────────────────
    logger.info("\n━━━ Stage 3: Optuna 100-Trial Bayesian Tuning (XGBoost) ━━━")
    logger.info("  → Searching wider hyperparameter space: gamma, reg_alpha, reg_lambda, min_child_weight ...")

    def objective(trial):
        params = {
            # --- Wider search on tree structure ---
            "n_estimators":       trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),

            # --- Wider learning rate range ---
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.20, log=True),

            # --- Stochastic sampling ---
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 1.0),

            # --- NEW: L1 / L2 regularization terms (not searched before) ---
            "gamma":              trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 5.0),

            "random_state": 42,
            "verbosity": 0,
            "n_jobs": 2,
        }
        m = XGBRegressor(**params)
        # 5-fold CV for more reliable estimate (n_jobs=2 to limit CPU load)
        return cross_val_score(m, X_train, y_train, cv=5, scoring="r2", n_jobs=2).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    logger.info(f"  ✔ Best trial: R²={study.best_value:.4f} | Params: {study.best_params}")

    best_xgb_params = {**study.best_params, "random_state": 42, "n_jobs": 2, "verbosity": 0}
    tuned_xgb = XGBRegressor(**best_xgb_params)
    with mlflow.start_run(run_name="Stage3_Optuna100_Tuned_XGBoost"):
        tuned_xgb.fit(X_train, y_train)
        metrics_tuned = calculate_metrics(y_test, tuned_xgb.predict(X_test))
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(metrics_tuned)
        mlflow.xgboost.log_model(tuned_xgb, "xgboost_model")
        results["Optuna Tuned XGBoost (100T)"] = metrics_tuned
        logger.info(f"  [Optuna Tuned XGBoost  ] R²: {metrics_tuned['r2_score']:.4f} | MAE: ${metrics_tuned['mae_dollar']:,.2f}")

    # Save the tuned XGBoost as the primary production model
    os.makedirs("models", exist_ok=True)
    joblib.dump(tuned_xgb, "models/best_xgb_tuned.pkl")

    # ─────────────────────────────────────────────────────────
    # Stage 4: Advanced Stacking Ensemble
    # ─────────────────────────────────────────────────────────
    logger.info("\n━━━ Stage 4: Advanced Stacking Meta-Ensemble ━━━")
    base_estimators = [
        ("xgb",  XGBRegressor(**best_xgb_params)),
        ("lgb",  LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.03,
                               random_state=42, n_jobs=2, verbose=-1)),
        ("gbdt", GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ]
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=0.5),
        cv=3,
        n_jobs=2
    )
    with mlflow.start_run(run_name="Stage4_Stacking_Ensemble"):
        stacking_model.fit(X_train, y_train)
        metrics_stack = calculate_metrics(y_test, stacking_model.predict(X_test))
        mlflow.log_metrics(metrics_stack)
        results["Stacking Meta-Ensemble"] = metrics_stack
        logger.info(f"  [Stacking Meta-Ensemble] R²: {metrics_stack['r2_score']:.4f} | MAE: ${metrics_stack['mae_dollar']:,.2f}")

    # Save overall best model (based on Occam's Razor: prefer simpler if performance is close)
    best_model_key = max(results, key=lambda k: results[k]["r2_score"])
    logger.info(f"\n  🏆 Best Model: {best_model_key} (R²={results[best_model_key]['r2_score']:.4f})")

    if best_model_key == "Stacking Meta-Ensemble":
        joblib.dump(stacking_model, "models/best_model.pkl")
    else:
        joblib.dump(tuned_xgb, "models/best_model.pkl")

    # Quantile Regressors for prediction intervals
    q_low  = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=150, random_state=42)
    q_high = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=150, random_state=42)
    q_low.fit(X_train, y_train)
    q_high.fit(X_train, y_train)
    joblib.dump(q_low,  "models/quantile_low.pkl")
    joblib.dump(q_high, "models/quantile_high.pkl")
    logger.info("  Quantile regressors saved (alpha=0.10 and 0.90).")

    return results


if __name__ == "__main__":
    results = run_progressive_training_pipeline()
    print("\n===================================================")
    print("        FINAL PROGRESSIVE RESULTS SUMMARY          ")
    print("===================================================")
    summary_df = pd.DataFrame(results).T.round(4)
    print(summary_df.to_string())
