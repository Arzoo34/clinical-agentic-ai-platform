"""Flask web UI for the AI-Driven Clinical Trial Intelligence Platform."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template

# Ensure src package is importable when running `flask run`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import config  # noqa: E402


def _load_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def create_app() -> Flask:
    app = Flask(__name__)

    predictions = _load_table(config.PREDICTIONS_CSV)
    insights = _load_table(config.INSIGHTS_CSV)
    metrics = {}
    if Path(config.MODEL_METRICS_JSON).exists():
        with open(config.MODEL_METRICS_JSON, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    def _top_insights(n: int = 5):
        if insights.empty:
            return []
        return (
            insights.sort_values("risk_probability", ascending=False)
            .head(n)[["site_id", "country", "risk_probability", "ai_insight", "alerts"]]
            .to_dict(orient="records")
        )

    @app.route("/")
    def home():
        return render_template("home.html", top_insights=_top_insights())

    @app.route("/architecture")
    def architecture():
        return render_template("architecture.html")

    @app.route("/workflow")
    def workflow():
        return render_template("workflow.html")

    @app.route("/results")
    def results():
        return render_template(
            "results.html",
            metrics=metrics,
            predictions_present=not predictions.empty,
            predictions_sample=predictions.head(50).to_dict(orient="records") if not predictions.empty else [],
        )

    @app.route("/dashboard")
    def dashboard():
        # Replace with actual Power BI embed URL or local report link
        power_bi_url = "https://app.powerbi.com/links/your-powerbi-report"
        return render_template("dashboard.html", power_bi_url=power_bi_url, insights=_top_insights())

    @app.route("/api/insights")
    def api_insights():
        return jsonify(_top_insights(20))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

