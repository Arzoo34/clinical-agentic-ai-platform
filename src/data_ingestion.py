"""
Data ingestion helpers: discover studies and load Excel sources into Spark DataFrames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from .config import RAW_DATA_BASE, SPARK_APP_NAME, SPARK_MASTER


def create_spark_session(
    app_name: str = SPARK_APP_NAME, master: str = SPARK_MASTER
) -> SparkSession:
    """Create a local Spark session with Arrow enabled for faster conversions."""
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def discover_study_dirs(raw_base: Path = RAW_DATA_BASE) -> List[Path]:
    """Return all study folders that contain anonymized CPID Excel inputs."""
    if not raw_base.exists():
        return []
    return sorted([p for p in raw_base.iterdir() if p.is_dir()])


def _standardize_columns(pdf: pd.DataFrame) -> pd.DataFrame:
    """Clean column headers: trim, replace whitespace/punctuation with snake_case."""
    pdf = pdf.copy()
    pdf.columns = (
        pdf.columns.astype(str)
        .str.replace(r"\n", " ", regex=True)
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    # Drop empty/unnamed columns
    pdf = pdf.loc[:, ~pdf.columns.str.contains(r"^unnamed", case=False)]
    return pdf


def _read_excel(path: Path, sheet: int | str = 0, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read an Excel sheet into a pandas DataFrame and standardize columns."""
    pdf = pd.read_excel(path, sheet_name=sheet, nrows=nrows)
    return _standardize_columns(pdf)


def _to_spark(spark: SparkSession, pdf: pd.DataFrame) -> DataFrame:
    """Convert pandas DataFrame to Spark DataFrame."""
    return spark.createDataFrame(pdf)


def _find_file(study_dir: Path, keyword: str) -> Optional[Path]:
    """Find the first Excel file whose name contains the keyword (case-insensitive)."""
    for path in study_dir.glob("*.xlsx"):
        if keyword.lower() in path.name.lower():
            return path
    return None


def load_subject_metrics(spark: SparkSession, study_dir: Path) -> Optional[DataFrame]:
    """Load the subject-level metrics table to map subjects to sites."""
    path = _find_file(study_dir, "cpid_edc_metrics")
    if not path:
        return None
    pdf = _read_excel(path, sheet=0)
    keep = [
        "project_name",
        "region",
        "country",
        "site_id",
        "site",
        "site_number",
        "site_name",
        "siteid",
        "subject_id",
        "subject",
        "subject_name",
        "latest_visit_sv_source_rave_edc_bo4",
        "subject_status_source_primary_form",
    ]
    pdf = pdf[[c for c in pdf.columns if c in keep]]
    pdf = pdf.rename(
        columns={
            "site": "site_id",
            "site_number": "site_id",
            "siteid": "site_id",
            "subject": "subject_id",
            "subject_name": "subject_id",
        }
    )
    pdf = pdf.dropna(subset=["site_id", "subject_id"])
    return _to_spark(spark, pdf)


def load_open_issues(spark: SparkSession, study_dir: Path) -> Optional[DataFrame]:
    path = _find_file(study_dir, "compiled_edrr")
    if not path:
        return None
    pdf = _read_excel(path, sheet=0)
    # Expect columns: study, subject, total_open_issue_count_per_subject
    rename_map = {
        "subject": "subject_id",
        "total_open_issue_count_per_subject": "open_issue_count",
    }
    pdf = pdf.rename(columns=rename_map)
    pdf["open_issue_count"] = pd.to_numeric(pdf.get("open_issue_count"), errors="coerce").fillna(0)
    pdf = pdf.dropna(subset=["subject_id"])
    return _to_spark(spark, pdf[["subject_id", "open_issue_count"]])


def load_sae(spark: SparkSession, study_dir: Path) -> Optional[DataFrame]:
    path = _find_file(study_dir, "sae")
    if not path:
        return None
    pdf = _read_excel(path, sheet=0)
    rename_map = {"patient_id": "subject_id", "site": "site_id", "study_id": "study"}
    pdf = pdf.rename(columns=rename_map)
    cols = [c for c in ["subject_id", "site_id", "country", "discrepancy_id"] if c in pdf.columns]
    if not cols:
        return None
    pdf = pdf[cols].dropna(subset=["subject_id"])
    return _to_spark(spark, pdf)


def load_coding_backlog(spark: SparkSession, study_dir: Path) -> Optional[DataFrame]:
    path = _find_file(study_dir, "codingreport")
    if not path:
        return None
    pdf = _read_excel(path, sheet=0)
    pdf = pdf.rename(columns={"subject": "subject_id"})
    if "require_coding" not in pdf.columns:
        return None
    pdf["require_coding_flag"] = pdf["require_coding"].astype(str).str.lower().str.contains("yes")
    pdf = pdf[["subject_id", "require_coding_flag"]].dropna(subset=["subject_id"])
    return _to_spark(spark, pdf)


def load_missing_pages(spark: SparkSession, study_dir: Path) -> Optional[DataFrame]:
    path = _find_file(study_dir, "missing_pages")
    if not path:
        return None
    pdf = _read_excel(path, sheet=0)
    pdf = pdf.rename(columns={"site_number": "site_id", "site": "site_id", "subject_name": "subject_id"})
    cols = [c for c in ["subject_id", "site_id", "#_of_days_missing"] if c in pdf.columns]
    if not cols:
        return None
    pdf = pdf[cols].dropna(subset=["subject_id"])
    pdf = pdf.rename(columns={"#_of_days_missing": "days_missing"})
    if "days_missing" in pdf.columns:
        pdf["days_missing"] = pd.to_numeric(pdf["days_missing"], errors="coerce")
    return _to_spark(spark, pdf)


def load_missing_labs(spark: SparkSession, study_dir: Path) -> Optional[DataFrame]:
    path = _find_file(study_dir, "missing_lab")
    if not path:
        return None
    pdf = _read_excel(path, sheet=0)
    pdf = pdf.rename(columns={"site_number": "site_id", "site": "site_id", "subject": "subject_id"})
    cols = [c for c in ["subject_id", "site_id"] if c in pdf.columns]
    if not cols:
        return None
    pdf = pdf[cols].dropna(subset=["subject_id"])
    return _to_spark(spark, pdf)


def load_inactivated_forms(spark: SparkSession, study_dir: Path) -> Optional[DataFrame]:
    path = _find_file(study_dir, "inactivated")
    if not path:
        return None
    pdf = _read_excel(path, sheet=0)
    pdf = pdf.rename(columns={"study_site_number": "site_id", "site_number": "site_id", "subject": "subject_id"})
    cols = [c for c in ["subject_id", "site_id", "audit_action"] if c in pdf.columns]
    if not cols:
        return None
    pdf = pdf[cols].dropna(subset=["subject_id"])
    return _to_spark(spark, pdf)


def load_visit_projection(spark: SparkSession, study_dir: Path) -> Optional[DataFrame]:
    path = _find_file(study_dir, "visit projection")
    if not path:
        return None
    pdf = _read_excel(path, sheet=0)
    pdf = pdf.rename(columns={"site": "site_id", "subject": "subject_id"})
    if "days_outstanding" in pdf.columns:
        pdf["days_outstanding"] = pd.to_numeric(pdf["days_outstanding"], errors="coerce")
    cols = [c for c in ["subject_id", "site_id", "days_outstanding"] if c in pdf.columns]
    pdf = pdf[cols].dropna(subset=["subject_id"])
    return _to_spark(spark, pdf)


def load_all_sources(
    spark: SparkSession, study_dir: Path
) -> Dict[str, Optional[DataFrame]]:
    """Load all available sources for a study directory."""
    return {
        "subject_metrics": load_subject_metrics(spark, study_dir),
        "open_issues": load_open_issues(spark, study_dir),
        "sae": load_sae(spark, study_dir),
        "coding": load_coding_backlog(spark, study_dir),
        "missing_pages": load_missing_pages(spark, study_dir),
        "missing_labs": load_missing_labs(spark, study_dir),
        "inactivated_forms": load_inactivated_forms(spark, study_dir),
        "visit_projection": load_visit_projection(spark, study_dir),
    }

