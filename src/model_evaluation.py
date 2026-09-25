import os
import logging
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.data_preprocessing import prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_shap_explanations():
    """Generate global and local SHAP explanations for model interpretability."""
    X_train, X_test, y_train, y_test, preprocessor, _ = prepare_data()
    
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Run training first.")
        return

    best_xgb = joblib.load(model_path)

    num_cols = ["age", "bmi", "children", "bmi_smoker"]
    cat_cols = ["sex", "smoker", "region"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    feature_names = num_cols + list(cat_encoder.get_feature_names_out(cat_cols))

    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test)

    os.makedirs("reports/figures", exist_ok=True)

    # 1. Global Feature Importance Summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    summary_path = "reports/figures/shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot to {summary_path}")

    # 2. Local Sample Explanation Waterfall Plot
    plt.figure(figsize=(9, 5))
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test[0],
        feature_names=feature_names
    )
    shap.waterfall_plot(explanation, show=False)
    waterfall_path = "reports/figures/shap_waterfall.png"
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP waterfall plot to {waterfall_path}")

    return [summary_path, waterfall_path]

if __name__ == "__main__":
    generate_shap_explanations()
