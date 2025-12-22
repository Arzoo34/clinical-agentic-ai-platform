"""
Project configuration for the AI-Driven Clinical Trial Intelligence Platform.
"""
from pathlib import Path

# Root folder that contains the anonymized study Excel files.
# The zip archive was extracted into this location.
RAW_DATA_BASE = Path("QC Anonymized Study Files")

# Where cleaned analytics/ML outputs will be written for Power BI and the web app.
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Spark defaults.
SPARK_APP_NAME = "ClinicalTrialIntelligence"
SPARK_MASTER = "local[*]"

# Filenames for exported artifacts.
SITE_FEATURES_CSV = OUTPUT_DIR / "site_features.csv"
SITE_FEATURES_PARQUET = OUTPUT_DIR / "site_features.parquet"
PREDICTIONS_CSV = OUTPUT_DIR / "site_risk_predictions.csv"
INSIGHTS_CSV = OUTPUT_DIR / "site_ai_insights.csv"
MODEL_METRICS_JSON = OUTPUT_DIR / "model_metrics.json"

