"""
End-to-end orchestration: ingest, engineer features, train model, generate insights, and export artifacts.
"""
from __future__ import annotations

import json
from functools import reduce
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pyspark.sql import DataFrame, functions as F

from . import config
from .data_ingestion import create_spark_session, discover_study_dirs, load_all_sources
from .feature_engineering import build_site_features
from .insights import attach_ai_layers
from .ml_model import add_rule_based_labels, train_logistic_model


def _union_frames(frames: List[DataFrame]) -> DataFrame:
    return reduce(lambda a, b: a.unionByName(b), frames)


def run_pipeline(raw_base: Path = config.RAW_DATA_BASE) -> Dict[str, object]:
    spark = create_spark_session()
    study_dirs = discover_study_dirs(raw_base)
    if not study_dirs:
        raise FileNotFoundError(f"No study folders found under {raw_base}")

    site_frames: List[DataFrame] = []
    for study_dir in study_dirs:
        sources = load_all_sources(spark, study_dir)
        site_features = build_site_features(sources)
        if site_features is None:
            continue
        site_features = site_features.withColumn("study_folder", F.lit(study_dir.name))
        site_frames.append(site_features)

    if not site_frames:
        raise ValueError("No site-level features could be constructed from the provided data.")

    all_sites = _union_frames(site_frames).dropDuplicates(["site_id", "country", "project_name"])
    all_sites = add_rule_based_labels(all_sites)

    # Train model and get hold-out metrics
    holdout_preds, metrics, model = train_logistic_model(all_sites)

    # Apply model to full dataset
    full_preds = model.transform(all_sites)
    full_preds = full_preds.withColumn("risk_probability", full_preds["probability"].getItem(1))
    enriched = attach_ai_layers(full_preds)

    # Collect pandas outputs for CSV/Parquet exports (drop vector columns for clean CSV)
    site_features_pd = all_sites.toPandas()
    export_cols = [
        "site_id",
        "country",
        "region",
        "project_name",
        "study_folder",
        "open_issues",
        "sae_events",
        "coding_backlog",
        "missing_pages",
        "missing_labs",
        "inactivated_forms",
        "avg_missing_days",
        "avg_days_outstanding",
        "risk_score_rule",
        "risk_label",
        "risk_probability",
        "alert_flag",
        "alert_count",
        "alerts",
        "ai_insight",
    ]
    predictions_pd = enriched.select([c for c in export_cols if c in enriched.columns]).toPandas()
    insights_pd = predictions_pd[
        [
            "site_id",
            "country",
            "region",
            "project_name",
            "risk_probability",
            "risk_label",
            "alert_flag",
            "alerts",
            "ai_insight",
        ]
    ]

    config.SITE_FEATURES_CSV.parent.mkdir(exist_ok=True, parents=True)
    site_features_pd.to_csv(config.SITE_FEATURES_CSV, index=False)
    site_features_pd.to_parquet(config.SITE_FEATURES_PARQUET, index=False)
    predictions_pd.to_csv(config.PREDICTIONS_CSV, index=False)
    insights_pd.to_csv(config.INSIGHTS_CSV, index=False)

    with open(config.MODEL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "site_features": all_sites,
        "holdout_predictions": holdout_preds,
        "predictions_path": config.PREDICTIONS_CSV,
        "insights_path": config.INSIGHTS_CSV,
        "metrics": metrics,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print("Pipeline complete. Outputs:")
    for k, v in results.items():
        if isinstance(v, (str, Path)):
            print(f"- {k}: {v}")
        elif isinstance(v, dict):
            print(f"- {k}: {json.dumps(v, indent=2)}")

