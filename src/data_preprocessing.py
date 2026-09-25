import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.data_ingestion import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_eda_plots(df: pd.DataFrame, output_dir: str = "reports/figures") -> list:
    """Generate and save EDA plots for report embedding."""
    os.makedirs(output_dir, exist_ok=True)
    generated_figures = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["charges"], bins=40, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Raw Medical Charges ($)", fontsize=11, fontweight="bold")

    sns.histplot(np.log1p(df["charges"]), bins=40, kde=True, ax=axes[1], color="#27ae60")
    axes[1].set_title("Log-Transformed Charges: log1p(charges)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "charges_distribution.png")
    plt.savefig(dist_path, dpi=200)
    plt.close()
    generated_figures.append((dist_path, "Target Distribution: Raw vs Log-Transformed Charges"))

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker",
                    palette={"yes": "#e74c3c", "no": "#3498db"}, alpha=0.8, s=60)
    plt.axvline(30, color="gray", linestyle="--", alpha=0.7, label="Obesity Threshold (BMI=30)")
    plt.title("BMI vs Medical Charges by Smoking Status & Obesity Threshold", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    inter_path = os.path.join(output_dir, "smoker_bmi_interaction.png")
    plt.savefig(inter_path, dpi=200)
    plt.close()
    generated_figures.append((inter_path, "BMI vs Medical Charges Segmented by Smoker Status"))

    plt.figure(figsize=(8, 5))
    numeric_df = df.copy()
    numeric_df["smoker_code"] = (numeric_df["smoker"] == "yes").astype(int)
    numeric_df["bmi_smoker"] = numeric_df["bmi"] * numeric_df["smoker_code"]
    corr = numeric_df[["age", "bmi", "children", "smoker_code", "bmi_smoker", "charges"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=200)
    plt.close()
    generated_figures.append((corr_path, "Feature Correlation Heatmap with Target"))

    logger.info(f"Generated {len(generated_figures)} EDA plots in {output_dir}")
    return generated_figures


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced Feature Engineering (3 Improvements applied):
    1. Polynomial features: bmi^2, age^2, age*smoker
    2. BMI Risk Tier Ordinal (4-level) replacing binary 'obese'
    3. Existing high-signal interactions retained: bmi_smoker, obese_smoker, age_bmi
    """
    df = df.copy()
    smoker_flag = (df["smoker"] == "yes").astype(int)

    # --- Improvement 1: Polynomial Features ---
    df["bmi_sq"] = df["bmi"] ** 2                          # Quadratic BMI effect
    df["age_sq"] = df["age"] ** 2                          # Quadratic age effect
    df["age_smoker"] = df["age"] * smoker_flag             # Age penalty amplified for smokers

    # --- Improvement 2: BMI Risk Tier (4-level Ordinal) ---
    # 0=Normal/Underweight (<25), 1=Overweight (25-30), 2=Obese (30-35), 3=Morbidly Obese (>=35)
    df["bmi_risk_tier"] = pd.cut(
        df["bmi"],
        bins=[0, 25, 30, 35, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # --- Retained interactions ---
    df["bmi_smoker"] = df["bmi"] * smoker_flag
    df["obese_smoker"] = (df["bmi"] >= 30).astype(int) * smoker_flag  # Critical step function
    df["age_bmi"] = df["age"] * df["bmi"]

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Scikit-learn ColumnTransformer with StandardScaler on all numeric features
    and OneHotEncoder for categorical features.
    """
    numeric_features = [
        "age", "bmi", "children",
        "bmi_smoker", "obese_smoker", "age_bmi",
        "bmi_sq", "age_sq", "age_smoker",
        "bmi_risk_tier"
    ]
    categorical_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    return preprocessor


def prepare_data(config_path: str = "params.yaml"):
    """Full preprocessing pipeline: Load -> Feature Engineering -> Split -> Fit Pipeline -> Save."""
    with open(config_path, "r") as f:
        import yaml
        config = yaml.safe_load(f)

    df = load_raw_data(config_path)
    eda_figures = generate_eda_plots(df)

    df_feat = create_features(df)

    X = df_feat.drop(columns=["charges"])
    y = np.log1p(df_feat["charges"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info(f"Preprocessor saved. Train shape: {X_train_proc.shape} | Test shape: {X_test_proc.shape}")

    return X_train_proc, X_test_proc, y_train.values, y_test.values, preprocessor, eda_figures


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, proc, figs = prepare_data()
    print(f"\nEnhanced Preprocessing Complete:")
    print(f"X_train shape: {X_tr.shape}")
    print(f"X_test shape:  {X_te.shape}")
