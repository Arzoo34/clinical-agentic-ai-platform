"""
Interpretable ML training using PySpark MLlib (logistic regression by default).
"""
from __future__ import annotations

from typing import Dict, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, functions as F


RISK_FEATURES = [
    "open_issues",
    "sae_events",
    "coding_backlog",
    "missing_pages",
    "missing_labs",
    "inactivated_forms",
    "avg_missing_days",
    "avg_days_outstanding",
]


def add_rule_based_labels(df: DataFrame) -> DataFrame:
    """Add a heuristic risk score and binary label for supervised learning."""
    for col_name in RISK_FEATURES:
        if col_name not in df.columns:
            df = df.withColumn(col_name, F.lit(0.0))
    df = df.fillna(0)
    risk_score = (
        F.col("open_issues") * 0.3
        + F.col("sae_events") * 3.0
        + F.col("missing_pages") * 0.5
        + F.col("missing_labs") * 0.4
        + F.col("coding_backlog") * 0.4
        + F.col("inactivated_forms") * 0.6
        + F.col("avg_missing_days") * 0.1
        + F.col("avg_days_outstanding") * 0.1
    )
    df = df.withColumn("risk_score_rule", risk_score)
    df = df.withColumn(
        "risk_label",
        F.when(
            (F.col("sae_events") >= 1)
            | (F.col("open_issues") >= 5)
            | (F.col("missing_pages") >= 3)
            | (F.col("inactivated_forms") >= 1)
            | (risk_score >= 5),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    return df


def train_logistic_model(df: DataFrame) -> Tuple[DataFrame, Dict, Pipeline]:
    """Train a logistic regression classifier and return predictions, metrics, and the fitted pipeline."""
    usable_features = [c for c in RISK_FEATURES if c in df.columns]
    if not usable_features:
        raise ValueError("No usable features found for model training.")
    assembler = VectorAssembler(inputCols=usable_features, outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="risk_label", probabilityCol="probability")
    pipeline = Pipeline(stages=[assembler, lr])

    train_df, test_df = df.randomSplit([0.75, 0.25], seed=42)
    model = pipeline.fit(train_df)
    preds = model.transform(test_df)

    evaluator_roc = BinaryClassificationEvaluator(labelCol="risk_label", rawPredictionCol="rawPrediction")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="risk_label", predictionCol="prediction", metricName="accuracy")
    metrics = {
        "roc_auc": evaluator_roc.evaluate(preds),
        "accuracy": evaluator_acc.evaluate(preds),
        "train_rows": train_df.count(),
        "test_rows": test_df.count(),
        "features_used": usable_features,
    }

    preds = preds.withColumn("risk_probability", F.col("probability").getItem(1))
    return preds, metrics, model

