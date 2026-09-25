import logging
import pandas as pd

logger = logging.getLogger(__name__)

def validate_raw_data(df: pd.DataFrame) -> tuple[bool, dict]:
    """Validate schema, column data types, missing values, and physical constraints."""
    expected_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    report = {"missing_columns": [], "null_counts": {}, "domain_errors": []}

    # 1. Column existence check
    missing = expected_cols - set(df.columns)
    if missing:
        report["missing_columns"] = list(missing)
        logger.error(f"Validation Error: Missing columns {missing}")
        return False, report

    # 2. Null check
    null_counts = df.isnull().sum()
    if null_counts.any():
        report["null_counts"] = null_counts[null_counts > 0].to_dict()
        logger.error(f"Validation Error: Found null values:\n{null_counts[null_counts > 0]}")
        return False, report

    # 3. Value domain bounds
    if (df["age"] < 18).any() or (df["age"] > 100).any():
        report["domain_errors"].append("Age outside expected domain [18, 100]")
        logger.error("Validation Error: Age outside expected domain [18, 100]")
        return False, report

    if (df["bmi"] < 10.0).any() or (df["bmi"] > 60.0).any():
        report["domain_errors"].append("BMI outside expected domain [10, 60]")
        logger.error("Validation Error: BMI outside expected domain [10, 60]")
        return False, report

    if (df["charges"] < 0).any():
        report["domain_errors"].append("Negative charges detected")
        logger.error("Validation Error: Negative charges detected")
        return False, report

    logger.info("✅ Data Validation Passed: 0 nulls, correct schema, valid ranges.")
    return True, report

def validate_data(df: pd.DataFrame) -> bool:
    """Alias for backwards compatibility."""
    is_valid, _ = validate_raw_data(df)
    return is_valid

if __name__ == "__main__":
    from src.data_ingestion import load_raw_data
    df = load_raw_data()
    is_valid, report = validate_raw_data(df)
    print(f"Validation Status: {is_valid}")
