"""
Feature engineering: harmonize subject-level inputs to site-level operational metrics.
"""
from __future__ import annotations

from functools import reduce
from typing import Dict, Optional

from pyspark.sql import DataFrame, functions as F

from .config import OUTPUT_DIR


def _safe_join(base: DataFrame, other: Optional[DataFrame]) -> DataFrame:
    if other is None:
        return base
    return base.join(other, ["site_id", "country", "region", "project_name"], how="full")


def build_site_features(sources: Dict[str, Optional[DataFrame]]) -> Optional[DataFrame]:
    """
    Aggregate multiple subject-level sources into a single site-level feature table.
    Expected keys in sources: subject_metrics (required), open_issues, sae, coding,
    missing_pages, missing_labs, inactivated_forms, visit_projection.
    """
    subject_map = sources.get("subject_metrics")
    if subject_map is None:
        return None

    # Normalize identifiers to strings to avoid numeric parsing issues from Excel.
    subject_map = (
        subject_map.select(
            F.col("subject_id").cast("string").alias("subject_id"),
            F.coalesce(F.col("site_id"), F.col("site_number")).cast("string").alias("site_id"),
            "country",
            "region",
            "project_name",
        )
        .dropna(subset=["site_id", "subject_id"])
    )

    feature_tables = []

    # Open issues
    if sources.get("open_issues") is not None:
        df = (
            sources["open_issues"]
            .join(subject_map, "subject_id", "left")
            .groupBy("site_id", "country", "region", "project_name")
            .agg(F.sum("open_issue_count").alias("open_issues"))
        )
        feature_tables.append(df)

    # SAE / discrepancies
    if sources.get("sae") is not None:
        df = (
            sources["sae"]
            .join(subject_map, "subject_id", "left")
            .groupBy("site_id", "country", "region", "project_name")
            .agg(F.countDistinct("discrepancy_id").alias("sae_events"))
        )
        feature_tables.append(df)

    # Coding backlog
    if sources.get("coding") is not None:
        df = (
            sources["coding"]
            .join(subject_map, "subject_id", "left")
            .groupBy("site_id", "country", "region", "project_name")
            .agg(F.sum(F.col("require_coding_flag").cast("int")).alias("coding_backlog"))
        )
        feature_tables.append(df)

    # Missing pages
    if sources.get("missing_pages") is not None:
        df = (
            sources["missing_pages"]
            .join(subject_map, "subject_id", "left")
            .groupBy("site_id", "country", "region", "project_name")
            .agg(
                F.count("*").alias("missing_pages"),
                F.avg("days_missing").alias("avg_missing_days"),
            )
        )
        feature_tables.append(df)

    # Missing labs
    if sources.get("missing_labs") is not None:
        df = (
            sources["missing_labs"]
            .join(subject_map, "subject_id", "left")
            .groupBy("site_id", "country", "region", "project_name")
            .agg(F.count("*").alias("missing_labs"))
        )
        feature_tables.append(df)

    # Inactivated forms
    if sources.get("inactivated_forms") is not None:
        df = (
            sources["inactivated_forms"]
            .join(subject_map, "subject_id", "left")
            .groupBy("site_id", "country", "region", "project_name")
            .agg(F.count("*").alias("inactivated_forms"))
        )
        feature_tables.append(df)

    # Visit projection / delays
    if sources.get("visit_projection") is not None:
        df = (
            sources["visit_projection"]
            .join(subject_map, "subject_id", "left")
            .groupBy("site_id", "country", "region", "project_name")
            .agg(F.avg("days_outstanding").alias("avg_days_outstanding"))
        )
        feature_tables.append(df)

    if not feature_tables:
        return None

    # Full outer join across all feature tables
    base = reduce(_safe_join, feature_tables)

    # Fill null numeric columns with zero, keep identifiers.
    numeric_cols = [f.name for f in base.schema.fields if f.name not in {"site_id", "country", "region", "project_name"}]
    for col in numeric_cols:
        base = base.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

    return base

