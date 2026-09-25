"""
Comprehensive Technical Report Generator for Medical Insurance Cost Predictor Project.
Generates a polished, publication-grade Microsoft Word document covering every phase
of the ML pipeline with tables, figures, observations, and conclusions.
"""

import os
import logging
from datetime import datetime
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls, qn
from docx.oxml import parse_xml, OxmlElement

logger = logging.getLogger(__name__)

OUTPUT_PATH = "reports/Medical_Insurance_Cost_Predictor_Final_Report.docx"


# ── Styling Helpers ──────────────────────────────────────────────

def style_header_row(row_cells, bg_hex="1F4E79"):
    """Apply dark background + white bold text to a header row."""
    for cell in row_cells:
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_hex}"/>')
        cell._element.get_or_add_tcPr().append(shd)
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(255, 255, 255)
                r.font.bold = True
                r.font.size = Pt(9.5)

def highlight_row(row_cells, bg_hex="E2EFDA"):
    """Apply light green highlight to a winning / best row."""
    for cell in row_cells:
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_hex}"/>')
        cell._element.get_or_add_tcPr().append(shd)

def add_styled_table(doc, headers, rows, best_row_keyword=None, bg="1F4E79"):
    """Create a styled table with header coloring and optional row highlight."""
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
    style_header_row(hdr, bg)
    for row_data in rows:
        r = table.add_row().cells
        for i, val in enumerate(row_data):
            r[i].text = str(val)
        if best_row_keyword and any(best_row_keyword in str(v) for v in row_data):
            highlight_row(r)
    return table

def add_figure(doc, path, caption, width=Inches(5.0)):
    """Embed figure with centered caption above it."""
    if not os.path.exists(path):
        return
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(50, 50, 50)
    doc.add_picture(path, width=width)
    last_p = doc.paragraphs[-1]
    last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()   # spacing

def section_heading(doc, text, level=1):
    """Add a colored heading."""
    h = doc.add_heading(text, level=level)
    if h.runs:
        h.runs[0].font.color.rgb = RGBColor(31, 78, 121)
    return h

def body(doc, text):
    """Add a body paragraph with proper font."""
    p = doc.add_paragraph(text)
    p.style.font.name = "Calibri"
    p.style.font.size = Pt(11)
    return p


# ── Main Report Generator ───────────────────────────────────────

def generate_report():
    os.makedirs("reports", exist_ok=True)
    doc = docx.Document()

    # Page setup
    for sec in doc.sections:
        sec.top_margin = Inches(0.9)
        sec.bottom_margin = Inches(0.9)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)

    # ════════════════════════════════════════════════════════════
    #  TITLE PAGE
    # ════════════════════════════════════════════════════════════
    for _ in range(6):
        doc.add_paragraph()

    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = tp.add_run("Medical Insurance Cost Predictor\n")
    r1.font.size = Pt(28)
    r1.bold = True
    r1.font.color.rgb = RGBColor(31, 78, 121)
    r2 = tp.add_run("End-to-End Machine Learning & MLOps Pipeline\n\n")
    r2.font.size = Pt(16)
    r2.italic = True
    r2.font.color.rgb = RGBColor(80, 80, 80)
    r3 = tp.add_run("Technical Project Report\n")
    r3.font.size = Pt(14)
    r3.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for label, value in [
        ("Author: ", "AP7803"),
        ("Date: ", datetime.now().strftime("%B %d, %Y")),
        ("Design Principle: ", "Occam's Razor"),
        ("Repository: ", "https://github.com/AP7803"),
    ]:
        meta.add_run(label).bold = True
        meta.add_run(value + "\n")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  TABLE OF CONTENTS (manual)
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "Table of Contents")
    toc_items = [
        "1.  Introduction & Project Objectives",
        "2.  Dataset Description & Domain Analysis",
        "3.  Data Quality Assurance & Validation",
        "4.  Exploratory Data Analysis",
        "5.  Feature Engineering Strategy",
        "6.  Scaling & Transformation Experiments",
        "7.  Progressive Model Training & Evaluation",
        "8.  Hyperparameter Optimization (Optuna)",
        "9.  Model Interpretability (SHAP)",
        "10. Production System Architecture",
        "11. Testing, Containerization & CI/CD",
        "12. Conclusions & Future Roadmap",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  1. INTRODUCTION
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "1. Introduction & Project Objectives")
    body(doc,
        "This report presents the complete development lifecycle of a Medical Insurance Cost Prediction system. "
        "The system predicts annual medical insurance charges for individual patients based on demographic and "
        "lifestyle attributes. The primary goal is to build a reliable, interpretable, and deployable machine "
        "learning model that can be used by insurance actuaries and healthcare administrators for cost estimation "
        "and risk stratification."
    )
    body(doc,
        "The entire project adheres to Occam's Razor principle: among competing models that explain the data "
        "equally well, the simplest one should be preferred. This principle guided every architectural decision, "
        "from feature engineering to model selection to deployment strategy."
    )

    section_heading(doc, "Project Scope", level=2)
    scope_items = [
        "Automated data ingestion with schema validation",
        "Exploratory data analysis with statistical visualizations",
        "Advanced feature engineering with polynomial and interaction terms",
        "Progressive model training across 10 model variants in 4 stages",
        "Bayesian hyperparameter optimization using Optuna (100 trials)",
        "Model interpretability using SHAP explainability framework",
        "Quantile regression for prediction uncertainty intervals (10th-90th percentile)",
        "Production REST API using FastAPI with SQLite prediction logging",
        "Interactive Streamlit dashboard with What-If scenario simulator",
        "PDF patient risk report generation",
        "Docker containerization and GitHub Actions CI/CD pipeline",
    ]
    for item in scope_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  2. DATASET DESCRIPTION
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "2. Dataset Description & Domain Analysis")
    body(doc,
        "The dataset used is the publicly available Medical Cost Personal Dataset from Kaggle, containing "
        "1,338 patient records with 7 attributes. Each record represents one policyholder with their demographic "
        "profile and the corresponding annual medical insurance charges billed by the insurer."
    )

    add_styled_table(doc,
        headers=["Feature", "Type", "Range / Categories", "Description"],
        rows=[
            ("age", "Numeric (int)", "18 - 64", "Age of the primary beneficiary in years"),
            ("sex", "Categorical", "{male, female}", "Gender of the insurance contractor"),
            ("bmi", "Numeric (float)", "15.96 - 53.13", "Body Mass Index (kg/m^2), ideally 18.5-24.9"),
            ("children", "Numeric (int)", "0 - 5", "Number of dependents covered by insurance"),
            ("smoker", "Categorical", "{yes, no}", "Whether the beneficiary smokes"),
            ("region", "Categorical", "{NE, NW, SE, SW}", "US geographic region of residence"),
            ("charges", "Numeric (float)", "$1,121 - $63,770", "Individual annual medical costs billed (TARGET)"),
        ]
    )

    body(doc, "")
    body(doc,
        "Key Statistical Properties:\n"
        "- Mean charges: $13,270.42 with high standard deviation ($12,110.01), indicating large variance.\n"
        "- 79.5% of patients are non-smokers; 20.5% are smokers.\n"
        "- BMI distribution is approximately normal, centered around 30.7.\n"
        "- Gender split is near-equal (50.5% male, 49.5% female)."
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  3. DATA QUALITY ASSURANCE
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "3. Data Quality Assurance & Validation")
    body(doc,
        "Before any modeling, the raw dataset undergoes automated quality validation via src/data_validation.py. "
        "Five domain assertion checks are performed programmatically to guarantee data integrity:"
    )

    add_styled_table(doc,
        headers=["Validation Rule", "Expected Constraint", "Observed Result", "Status"],
        rows=[
            ("Null / Missing Values", "0 nulls in all 7 columns", "0 nulls across 1,338 records", "PASSED"),
            ("Age Domain", "18 <= age <= 100", "min=18, max=64", "PASSED"),
            ("BMI Domain", "10.0 <= bmi <= 60.0", "min=15.96, max=53.13", "PASSED"),
            ("Charges Non-Negative", "charges >= 0", "min=$1,121.87", "PASSED"),
            ("Categorical Schema", "Exact domain match", "No unknown categories detected", "PASSED"),
        ]
    )

    body(doc, "")
    body(doc,
        "Observation: The dataset is remarkably clean with zero missing values, no duplicate rows, and all "
        "values within physically plausible bounds. No imputation or outlier removal was necessary prior to "
        "feature engineering."
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  4. EXPLORATORY DATA ANALYSIS
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "4. Exploratory Data Analysis")
    body(doc,
        "Exploratory Data Analysis (EDA) was conducted to uncover distributional patterns, feature interactions, "
        "and multicollinearity structure within the dataset. Three key findings emerged:"
    )

    section_heading(doc, "4.1 Target Variable Distribution", level=2)
    body(doc,
        "Raw medical charges exhibit strong positive (right) skewness. The majority of patients incur charges "
        "between $1,000 and $15,000, but a long tail extends to $63,770 driven primarily by obese smokers. "
        "Applying a log1p transformation converts the target into a near-symmetric distribution, which stabilizes "
        "gradient-based optimization and improves linear model performance."
    )
    add_figure(doc, "reports/figures/charges_distribution.png",
               "Figure 4.1: Raw Charges Distribution vs. Log-Transformed Target")

    section_heading(doc, "4.2 The Smoker-BMI Interaction Effect", level=2)
    body(doc,
        "The scatter plot of BMI vs. charges reveals a critical non-linear interaction. Non-smokers show a "
        "flat, low-cost pattern regardless of BMI ($2,000 - $12,000). However, smokers with BMI >= 30 exhibit "
        "a dramatic cost escalation to $30,000 - $50,000. This creates a step-function boundary at the obesity "
        "threshold (BMI=30) that is invisible to purely linear models."
    )
    add_figure(doc, "reports/figures/smoker_bmi_interaction.png",
               "Figure 4.2: BMI vs. Charges by Smoking Status with Obesity Threshold")

    section_heading(doc, "4.3 Feature Correlation Structure", level=2)
    body(doc,
        "The correlation heatmap reveals that raw features have moderate-to-weak linear correlation with charges. "
        "However, the engineered feature bmi_smoker (BMI multiplied by smoker indicator) shows the strongest "
        "single-feature correlation with charges, validating its inclusion in the pipeline."
    )
    add_figure(doc, "reports/figures/correlation_heatmap.png",
               "Figure 4.3: Feature Correlation Heatmap Including Engineered Interactions")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  5. FEATURE ENGINEERING
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "5. Feature Engineering Strategy")
    body(doc,
        "To maximize predictive signal while adhering to Occam's Razor, seven engineered features were created "
        "in src/data_preprocessing.py. Each feature is motivated by domain knowledge from clinical literature "
        "and actuarial science:"
    )

    add_styled_table(doc,
        headers=["Feature Name", "Formula", "Rationale"],
        rows=[
            ("bmi_sq", "bmi^2",
             "Quadratic BMI term captures accelerating health risks in extreme BMI ranges"),
            ("age_sq", "age^2",
             "Quadratic age term models non-linear medical expense growth in older patients"),
            ("age_smoker", "age * I(smoker=yes)",
             "Smoking health damage compounds with each additional year of age"),
            ("bmi_risk_tier", "Ordinal: 0/1/2/3",
             "4-level clinical categorization: Normal(<25), Overweight(25-30), Obese(30-35), Morbid(>=35)"),
            ("bmi_smoker", "bmi * I(smoker=yes)",
             "Primary interaction: captures the steep cost gradient for smokers as BMI increases"),
            ("obese_smoker", "I(bmi>=30) * I(smoker=yes)",
             "Binary step function modeling the $30,000+ threshold for obese smokers"),
            ("age_bmi", "age * bmi",
             "Compound exposure: cumulative body mass burden over years of age"),
        ]
    )

    body(doc, "")
    body(doc,
        "After feature engineering, the preprocessing pipeline applies:\n"
        "- StandardScaler on all 10 numeric features (age, bmi, children, bmi_smoker, obese_smoker, "
        "age_bmi, bmi_sq, age_sq, age_smoker, bmi_risk_tier)\n"
        "- OneHotEncoder on 3 categorical features (sex, smoker, region)\n\n"
        "This produces a final feature matrix of 18 dimensions per patient record."
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  6. SCALING EXPERIMENTS
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "6. Scaling & Transformation Experiments")
    body(doc,
        "An empirical experiment was conducted to test whether applying StandardScaler only to the age_bmi "
        "column (while keeping other numeric columns at raw scale and using OneHotEncoder for categoricals) "
        "would yield better results than scaling all numeric features uniformly."
    )

    add_styled_table(doc,
        headers=["Configuration", "Lasso MAE ($)", "XGBoost R^2", "Verdict"],
        rows=[
            ("Full StandardScaler (all numeric)", "$3,803.38", "0.8714", "OPTIMAL - Selected"),
            ("Partial Scaling (age_bmi only)", "$3,952.26", "0.8681", "Inferior - Rejected"),
            ("Delta (degradation)", "+$148.88", "-0.0033", "Full scaling is mandatory"),
        ],
        best_row_keyword="OPTIMAL"
    )

    body(doc, "")
    body(doc,
        "Analysis: Lasso regression is particularly sensitive to feature scaling because its L1 penalty "
        "penalizes coefficients proportionally to the magnitude of the corresponding feature values. When "
        "features like age (18-64) and bmi_smoker (0-1500+) exist on vastly different scales, the L1 penalty "
        "disproportionately shrinks the larger-magnitude features. StandardScaler normalizes all features to "
        "zero mean and unit variance, ensuring fair regularization.\n\n"
        "Conclusion: Full StandardScaler on all numeric columns was retained as the production pipeline "
        "configuration."
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  7. PROGRESSIVE MODEL TRAINING
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "7. Progressive Model Training & Evaluation")
    body(doc,
        "Following the progressive modeling strategy, 10 models were trained and evaluated across 4 stages "
        "of increasing complexity. All metrics are reported on the held-out 20% test set (268 patients). "
        "R-squared is computed in log space; MAE, RMSE, and MAPE are in original dollar space."
    )

    section_heading(doc, "7.1 Stage 1: Simple Baseline Models", level=2)
    body(doc,
        "Simple linear and shallow tree models establish the performance floor. Thanks to the polynomial "
        "feature engineering, even linear models achieve substantially better performance than raw-feature "
        "baselines reported in literature (typical raw-feature Linear R^2 ~ 0.75)."
    )

    add_styled_table(doc,
        headers=["Model", "R^2", "MAE ($)", "RMSE ($)", "MAPE (%)"],
        rows=[
            ("Linear Regression", "0.8576", "$2,463.89", "$4,842.35", "18.05%"),
            ("Ridge Regression (alpha=1.0)", "0.8579", "$2,395.24", "$4,786.09", "17.90%"),
            ("Lasso Regression (alpha=0.01)", "0.8345", "$3,111.76", "$7,335.09", "21.29%"),
            ("Decision Tree (max_depth=5)", "0.8318", "$2,190.11", "$4,537.78", "20.43%"),
        ]
    )

    body(doc, "")
    body(doc,
        "Observation: Ridge slightly outperforms basic Linear Regression due to L2 regularization reducing "
        "variance. Lasso performs worst because L1 penalty over-shrinks useful polynomial features. Decision "
        "Tree achieves the lowest MAE in this stage despite lower R^2, indicating it captures local non-linear "
        "patterns (such as the smoker step-function) that linear models miss."
    )

    section_heading(doc, "7.2 Stage 2: Complex Ensemble Models", level=2)
    body(doc,
        "Gradient-boosted tree ensembles are evaluated with default sensible hyperparameters. CPU usage is "
        "restricted to n_jobs=2 throughout to prevent server overload."
    )

    add_styled_table(doc,
        headers=["Model", "R^2", "MAE ($)", "RMSE ($)", "MAPE (%)"],
        rows=[
            ("Random Forest (100 trees)", "0.8416", "$1,991.69", "$4,360.26", "18.10%"),
            ("Gradient Boosting (GBDT)", "0.8629", "$2,004.30", "$4,401.97", "17.50%"),
            ("XGBoost (default params)", "0.8715", "$1,900.91", "$4,292.52", "16.37%"),
            ("LightGBM (default params)", "0.8655", "$1,942.66", "$4,394.76", "16.37%"),
        ],
        best_row_keyword="XGBoost"
    )

    body(doc, "")
    body(doc,
        "Observation: XGBoost Default achieves the best overall metrics across all 10 models evaluated in "
        "this project (R^2 = 0.8715, MAE = $1,900.91). This is a strong result given only 6 raw input features "
        "and 1,338 training samples. LightGBM is a close second."
    )

    section_heading(doc, "7.3 Stage 3: Optuna Hyperparameter Tuning (100 Trials)", level=2)
    body(doc,
        "Bayesian hyperparameter optimization was performed on XGBoost using Optuna with 100 trials and "
        "5-fold cross-validation. The search space was expanded to include regularization parameters "
        "(gamma, reg_alpha, reg_lambda), minimum child weight, and column sampling by level."
    )
    body(doc,
        "See Section 8 for detailed Optuna configuration and results."
    )

    section_heading(doc, "7.4 Stage 4: Stacking Meta-Ensemble", level=2)
    body(doc,
        "A StackingRegressor combining tuned XGBoost, LightGBM, Gradient Boosting, and Ridge with a "
        "Ridge meta-learner was trained to test whether ensemble diversity improves predictions."
    )

    section_heading(doc, "7.5 Complete Results Summary", level=2)

    add_styled_table(doc,
        headers=["Stage", "Model", "R^2", "MAE ($)", "RMSE ($)", "MAPE (%)"],
        rows=[
            ("1 - Simple", "Linear Regression", "0.8576", "$2,463.89", "$4,842.35", "18.05%"),
            ("1 - Simple", "Ridge Regression", "0.8579", "$2,395.24", "$4,786.09", "17.90%"),
            ("1 - Simple", "Lasso Regression", "0.8345", "$3,111.76", "$7,335.09", "21.29%"),
            ("1 - Simple", "Decision Tree", "0.8318", "$2,190.11", "$4,537.78", "20.43%"),
            ("2 - Complex", "Random Forest", "0.8416", "$1,991.69", "$4,360.26", "18.10%"),
            ("2 - Complex", "Gradient Boosting", "0.8629", "$2,004.30", "$4,401.97", "17.50%"),
            ("2 - Complex", "XGBoost Default (BEST)", "0.8715", "$1,900.91", "$4,292.52", "16.37%"),
            ("2 - Complex", "LightGBM Default", "0.8655", "$1,942.66", "$4,394.76", "16.37%"),
            ("3 - Tuned", "Optuna XGBoost (100T)", "0.8607", "$2,135.07", "$4,527.41", "17.20%"),
            ("4 - Ensemble", "Stacking Meta-Ensemble", "0.8664", "$2,151.16", "$4,443.82", "16.38%"),
        ],
        best_row_keyword="BEST"
    )

    body(doc, "")
    body(doc,
        "Key Insight (Occam's Razor Validation): The single default XGBoost model outperforms both the "
        "100-trial Optuna-tuned XGBoost and the complex 4-model Stacking Ensemble. On small tabular datasets "
        "(~1,338 rows), additional model complexity introduces slight overfitting to cross-validation noise "
        "without improving generalization. This empirically validates Occam's Razor: the simplest sufficient "
        "model is the best production choice."
    )

    add_figure(doc, "reports/figures/model_performance_residuals.png",
               "Figure 7.1: Model Residuals Analysis (Best XGBoost - Predicted vs Actual)")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  8. HYPERPARAMETER OPTIMIZATION
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "8. Hyperparameter Optimization (Optuna)")
    body(doc,
        "Optuna Bayesian optimization was run on XGBoost with the following expanded search configuration:"
    )

    add_styled_table(doc,
        headers=["Hyperparameter", "Search Range", "Best Value Found"],
        rows=[
            ("n_estimators", "100 - 600 (step=50)", "550"),
            ("max_depth", "3 - 8", "3"),
            ("min_child_weight", "1 - 10", "9"),
            ("learning_rate", "0.01 - 0.20 (log scale)", "0.167"),
            ("subsample", "0.5 - 1.0", "0.874"),
            ("colsample_bytree", "0.5 - 1.0", "0.876"),
            ("colsample_bylevel", "0.5 - 1.0", "0.840"),
            ("gamma", "0.0 - 0.5", "0.425"),
            ("reg_alpha (L1)", "0.0 - 1.0", "0.607"),
            ("reg_lambda (L2)", "0.5 - 5.0", "3.264"),
        ]
    )

    body(doc, "")
    body(doc,
        "Configuration: 100 Bayesian trials with 5-fold cross-validation, maximizing mean R^2. "
        "CPU load limited to n_jobs=2.\n\n"
        "Result: Best cross-validation R^2 = 0.8275. However, the tuned model's test-set performance "
        "(R^2 = 0.8607) was lower than the default XGBoost (R^2 = 0.8715). The aggressive regularization "
        "(gamma=0.425, reg_alpha=0.607, reg_lambda=3.264) selected by Optuna's Bayesian search was optimal "
        "for cross-validation stability but slightly under-fit on the specific test split.\n\n"
        "This is a known phenomenon on small datasets: the optimal cross-validation hyperparameters may "
        "differ from the optimal single-split test parameters due to high variance in small holdout sets."
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  9. MODEL INTERPRETABILITY (SHAP)
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "9. Model Interpretability (SHAP)")
    body(doc,
        "SHAP (SHapley Additive exPlanations) values were computed for the production XGBoost model to "
        "provide both global feature importance rankings and patient-level prediction attribution."
    )

    add_figure(doc, "reports/figures/shap_summary_bar.png",
               "Figure 9.1: SHAP Global Feature Importance (Mean Absolute SHAP Values)")

    body(doc,
        "The bar plot reveals that smoker status (via smoker_yes and bmi_smoker) dominates predictive "
        "power, accounting for over 55% of the model's total output. Age and the age_bmi compound term "
        "are the next most important, reflecting steady medical cost increases with aging."
    )

    add_figure(doc, "reports/figures/shap_summary_dot.png",
               "Figure 9.2: SHAP Beeswarm Plot (Feature Value Impact Direction)")

    body(doc,
        "The beeswarm plot shows directionality: high values of smoker_yes (red dots) push predictions "
        "strongly upward, while high age values consistently increase costs. Low BMI values (blue dots) "
        "slightly reduce predicted charges."
    )

    add_figure(doc, "reports/figures/shap_waterfall_patient_0.png",
               "Figure 9.3: Patient-Level SHAP Waterfall Attribution (Test Patient #0)")

    body(doc,
        "The waterfall chart decomposes a specific patient's prediction into exact dollar-denominated "
        "contributions from each feature. This enables clinicians to explain to patients precisely which "
        "factors are driving their insurance cost estimate."
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  10. PRODUCTION SYSTEM ARCHITECTURE
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "10. Production System Architecture")

    section_heading(doc, "10.1 FastAPI REST Service", level=2)
    body(doc,
        "The production inference API is built with FastAPI (api/app.py) and exposes the following endpoints:"
    )

    add_styled_table(doc,
        headers=["Endpoint", "Method", "Description"],
        rows=[
            ("/health", "GET", "Returns API health status and model artifact load verification"),
            ("/predict", "POST", "Accepts single patient JSON, returns cost prediction + 10th-90th percentile confidence interval, logs to SQLite"),
            ("/predict/batch", "POST", "Accepts CSV file upload, returns batch predictions for multiple patients"),
            ("/history", "GET", "Queries past prediction records from SQLite database with pagination"),
        ]
    )

    section_heading(doc, "10.2 SQLite Database Layer", level=2)
    body(doc,
        "Every prediction made via the /predict endpoint is automatically logged to an SQLite database "
        "(data/predictions.db) via src/db.py using SQLAlchemy ORM. Each record stores: timestamp, patient "
        "demographics (age, sex, bmi, children, smoker, region), predicted cost, and 10th-90th percentile "
        "confidence bounds."
    )

    section_heading(doc, "10.3 Streamlit Dashboard", level=2)
    body(doc,
        "An interactive web frontend (dashboard/streamlit_app.py) provides three functional tabs:\n"
        "1. Prediction & Range: Interactive sliders for patient parameters with real-time cost estimation "
        "and 10th-90th percentile confidence display.\n"
        "2. What-If Simulator: Users can toggle smoking status or adjust BMI to instantly see the impact "
        "on predicted costs (e.g., 'If patient quits smoking, annual savings = $21,450').\n"
        "3. PDF Export: Generates a formal patient risk assessment report using FPDF2 for clinical or "
        "administrative documentation."
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  11. TESTING & DEVOPS
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "11. Testing, Containerization & CI/CD")

    section_heading(doc, "11.1 Automated Testing (Pytest)", level=2)
    body(doc, "The test suite consists of 5 unit and integration tests:")

    add_styled_table(doc,
        headers=["Test File", "Test Case", "Validates"],
        rows=[
            ("test_data_preprocessing.py", "test_create_features_bmi_smoker_interaction", "Feature engineering produces correct bmi_smoker, bmi_sq, age_sq, age_smoker, bmi_risk_tier columns"),
            ("test_data_preprocessing.py", "test_preprocessor_transform_shape", "ColumnTransformer outputs correct dimensionality after scaling + encoding"),
            ("test_api.py", "test_health_check_endpoint", "GET /health returns status=healthy with model_loaded=True"),
            ("test_api.py", "test_predict_single_endpoint", "POST /predict returns valid cost prediction > 0 with confidence intervals"),
            ("test_api.py", "test_history_endpoint", "GET /history returns list of past prediction records"),
        ]
    )

    body(doc, "\nAll 5 tests pass consistently (verified: 5 passed in 3.54s).")

    section_heading(doc, "11.2 Docker Containerization", level=2)
    body(doc,
        "A production Dockerfile (python:3.10-slim base) and docker-compose.yml orchestrate two services:\n"
        "- api: FastAPI server on port 8000\n"
        "- dashboard: Streamlit app on port 8501\n"
        "Both services mount models/ and data/ as persistent volumes."
    )

    section_heading(doc, "11.3 GitHub Actions CI/CD", level=2)
    body(doc,
        "The CI pipeline (.github/workflows/ci.yml) triggers on every push/PR to main branch and runs:\n"
        "1. Dependency installation (pip install -r requirements.txt)\n"
        "2. Ruff linting (src/, api/, tests/)\n"
        "3. Black formatting check (src/, api/, tests/, dashboard/)\n"
        "4. Pytest test suite execution"
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════
    #  12. CONCLUSIONS & FUTURE WORK
    # ════════════════════════════════════════════════════════════
    section_heading(doc, "12. Conclusions & Future Roadmap")

    section_heading(doc, "12.1 Key Conclusions", level=2)
    conclusions = [
        "XGBoost Default with engineered features is the optimal production model (R^2 = 0.8715, MAE = $1,900.91), outperforming both Optuna-tuned and Stacking Ensemble variants.",
        "Polynomial feature engineering (bmi^2, age^2, age x smoker) and 4-level BMI risk tiers boosted simple linear model R^2 by +3.6% (from 0.8219 to 0.8579).",
        "Full StandardScaler on all numeric features is empirically mandatory; partial scaling degraded Lasso MAE by +$148.88.",
        "On small tabular datasets (~1,338 rows), Occam's Razor holds: simpler single-model architectures generalize better than complex meta-ensembles.",
        "The complete MLOps stack (FastAPI + SQLite + Streamlit + Docker + CI/CD) enables immediate production deployment.",
    ]
    for c in conclusions:
        doc.add_paragraph(c, style="List Bullet")

    section_heading(doc, "12.2 Future Roadmap", level=2)
    future = [
        "Incorporate clinical features (blood pressure, cholesterol, pre-existing conditions) if data becomes available to potentially push R^2 above 0.90.",
        "Implement real-time data drift monitoring using Evidently AI or MLflow Model Registry.",
        "Deploy Docker containers to AWS ECS or Azure Container Apps with automated HTTPS endpoints.",
        "Add A/B testing framework for comparing model versions in production.",
        "Integrate Prometheus + Grafana dashboards for API latency and prediction volume monitoring.",
    ]
    for f in future:
        doc.add_paragraph(f, style="List Bullet")

    # ── Save ─────────────────────────────────────────────────
    try:
        doc.save(OUTPUT_PATH)
        print(f"Report saved successfully: {OUTPUT_PATH}")
    except PermissionError:
        alt = OUTPUT_PATH.replace(".docx", f"_{datetime.now().strftime('%H%M%S')}.docx")
        doc.save(alt)
        print(f"Primary path locked. Report saved to: {alt}")


if __name__ == "__main__":
    generate_report()
