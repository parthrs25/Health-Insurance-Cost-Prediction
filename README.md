# Medical Insurance Cost Predictor
## End-to-End Machine Learning & MLOps Pipeline

This repository contains a publication-grade, end-to-end Machine Learning pipeline and MLOps system that predicts annual medical insurance charges based on patient demographic and lifestyle attributes. The project is designed with a progressive model training philosophy under the guidance of the **Occam's Razor** design principle.

---

## 🚀 Key Features

*   **Data Validation & Quality Assurance**: Automated schema and clinical range checks on raw data using programmatic assertions.
*   **Feature Engineering**: Custom polynomial features ($BMI^2$, $Age^2$, $Age \times Smoker$), interaction terms ($BMI \times Smoker$), and clinical BMI risk tiers.
*   **Progressive Model Selection**: Evaluations across 10 different modeling architectures from simple linear baselines to advanced tree-based ensembles (XGBoost, LightGBM, Gradient Boosting).
*   **Hyperparameter Tuning**: 100-trial Bayesian hyperparameter search using Optuna with 5-fold cross-validation.
*   **Model Explainability (SHAP)**: Global and patient-level predictions explanation using SHAP values (Beeswarm, waterfall, and bar charts).
*   **Production Deployment ready**:
    *   **FastAPI REST Backend** with automated SQLite database logging of all predictions.
    *   **Streamlit Web Interface** with interactive What-If scenario simulations.
    *   **PDF Generation** for formal risk assessments.
    *   **Docker Containerization** for multi-service environments (`docker-compose`).
    *   **GitHub Actions CI/CD Pipeline** checking formatting (Black), linting (Ruff), and unit testing (Pytest).

---

## 📊 Modeling Results Summary

The modeling suite was run in log space for stabilizing gradients, and evaluated on a held-out 20% test set (268 patients). Metrics in original dollar space:

| Stage | Model | $R^2$ | MAE ($) | RMSE ($) | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stage 1 (Simple)** | Linear Regression | 0.8576 | $2,463.89 | $4,842.35 | 18.05% |
| | Ridge Regression | 0.8579 | $2,395.24 | $4,786.09 | 17.90% |
| | Lasso Regression | 0.8345 | $3,111.76 | $7,335.09 | 21.29% |
| | Decision Tree | 0.8318 | $2,190.11 | $4,537.78 | 20.43% |
| **Stage 2 (Complex)**| Random Forest | 0.8416 | $1,991.69 | $4,360.26 | 18.10% |
| | Gradient Boosting | 0.8629 | $2,004.30 | $4,401.97 | 17.50% |
| | **XGBoost Default (BEST)** | **0.8715** | **$1,900.91** | **$4,292.52** | **16.37%** |
| | LightGBM Default | 0.8655 | $1,942.66 | $4,394.76 | 16.37% |
| **Stage 3 (Tuned)** | Optuna XGBoost (100 Trials) | 0.8607 | $2,135.07 | $4,527.41 | 17.20% |
| **Stage 4 (Stacking)**| Stacking Meta-Ensemble | 0.8664 | $2,151.16 | $4,443.82 | 16.38% |

> 💡 **Occam's Razor Insight**: The single default XGBoost model outperformed both the tuned Optuna variant and the Stacking Ensemble on our tabular dataset. This empirically validated that the simplest sufficient model is the best production choice to avoid overfitting.

---

## 🛠️ Project Structure

```
.
├── .github/workflows/      # GitHub Actions CI configurations
├── api/                    # FastAPI REST Application
│   └── app.py              # Main REST API endpoints
├── dashboard/              # Streamlit Web App
│   ├── streamlit_app.py    # Main UI dashboard
│   └── pdf_generator.py    # PDF report creator
├── data/
│   ├── raw/                # Original insurance dataset
│   └── predictions.db      # SQLite database logging API prediction history
├── models/                 # Pretrained joblib pipelines and models
├── notebooks/              # Jupyter notebooks for EDA and modeling experiments
├── reports/
│   ├── figures/            # Visualizations (EDA, SHAP, residuals)
│   └── Medical_Insurance_Cost_Predictor_Final_Report.docx  # Full 12-section project report
├── src/                    # Core source codebase
│   ├── data_ingestion.py   # Raw data loader
│   ├── data_preprocessing.py # Feature engineering & transformations
│   ├── data_validation.py  # Quality validation checks
│   ├── model_training.py   # Progressive training pipeline
│   ├── report_builder.py   # Word document generator
│   └── db.py               # SQLite / SQLAlchemy helper
├── tests/                  # Pytest unit & integration test files
├── Dockerfile              # Docker container setup
├── docker-compose.yml      # Orchestration setup for API + Streamlit
├── requirements.txt        # Production python packages
└── README.md               # Project documentation (this file)
```

---

## ⚙️ Quick Start

### 1. Prerequisites
Make sure python `3.10+` and `virtualenv` are installed:
```bash
# Clone the repository
git clone https://github.com/AP7803/medical-cost-predictor.git
cd medical-cost-predictor

# Create & activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Pipelines
To preprocess data, retrain models, and generate the final report:
```bash
# Preprocess and Feature Engineer data
python -m src.data_preprocessing

# Run training, evaluation, and Optuna tuning
python -m src.model_training

# Generate SHAP explainability figures
python src/generate_shap_figures.py

# Build the publication-ready Word Report
python -m src.report_builder
```

### 3. Launch Services Locally
You can run both services natively on your system:
```bash
# Start FastAPI server (logs SQLite predictions to data/predictions.db)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Start the interactive Streamlit dashboard (separate terminal)
streamlit run dashboard/streamlit_app.py
```

### 4. Running with Docker Compose
To build and run both the API and Streamlit Dashboard simultaneously inside isolated containers:
```bash
docker-compose up --build
```
*   FastAPI is exposed at: `http://localhost:8000`
*   Streamlit UI is exposed at: `http://localhost:8501`

---

## 🧪 Testing

To run the unit/integration tests verifying the pipeline components, FastAPI endpoints, and prediction database:
```bash
pytest tests/
```
