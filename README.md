# AI-Driven Clinical Trial Intelligence Platform

End-to-end, resume-grade project that ingests anonymized study datasets, harmonizes subject-to-site metrics with PySpark, trains an interpretable logistic regression model to predict site-level operational risk, generates plain-English insights, and serves results through a Flask website with Power BI-ready exports.

## Data (already provided)
- Raw inputs live under `QC Anonymized Study Files/` (unzipped from the supplied archives).
- Per-study Excel inputs: CPID EDC metrics, compiled EDRR (open issues), SAE dashboards, coding reports, missing pages/labs, inactivated forms, and visit projection trackers.
- The pipeline reads these directly; **raw files are never altered**.
- Column mapping (post-standardization): `project_name/region/country/site_id/subject_id` from CPID metrics; `total_open_issue_count_per_subject` → `open_issue_count`; SAE uses `discrepancy_id`, `patient_id` → `subject_id`, `site`; coding uses `require_coding`; missing pages uses `# of days missing`; visit tracker uses `# days outstanding`; inactivated forms uses `study_site_number/site_number` and `audit_action`.

## Architecture
- **Ingestion (PySpark + pandas):** `src/data_ingestion.py` standardizes column names and converts Excel sheets to Spark DataFrames.
- **Feature engineering:** `src/feature_engineering.py` maps subjects to sites and aggregates open issues, SAE counts, coding backlog, missing pages/labs, inactivated forms, and visit delays.
- **Risk labeling + ML:** `src/ml_model.py` applies rule-based labels then trains logistic regression for interpretable probabilities.
- **Generative explanations + agentic alerts:** `src/insights.py` turns metrics into English narratives and rule-based alert reasons.
- **Pipeline orchestrator:** `src/pipeline.py` runs end-to-end and writes Power BI-friendly CSV/Parquet.
- **Web app:** `app/main.py` (Flask) with pages for Home, Architecture, AI Workflow, Results, and Dashboard (embed/link Power BI).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run the data + ML pipeline
```bash
python -m src.pipeline
```
Outputs (Power BI-ready) are written to `outputs/`:
- `site_features.csv` / `site_features.parquet`: engineered site metrics + rule-based labels.
- `site_risk_predictions.csv`: model predictions, probabilities, alerts, insights.
- `site_ai_insights.csv`: concise insight feed for the UI/Power BI.
- `model_metrics.json`: ROC AUC, accuracy, dataset sizes, features used.

## Run the Flask website
```bash
set FLASK_APP=app.main
flask run
# then open http://127.0.0.1:5000
```
Pages:
- **Home:** problem/solution and top AI insights.
- **Architecture:** system overview.
- **AI Workflow:** step-by-step data & ML flow.
- **Results:** model quality and sample predictions.
- **Dashboard:** iframe placeholder to embed/publish your Power BI report + live insights.

## Power BI integration
- In Power BI Desktop, import `outputs/site_risk_predictions.csv` or `site_features.parquet`.
- Build visuals (risk trend, alert counts, SAE vs. backlog).
- Publish and replace the placeholder embed URL in `app/templates/dashboard.html`.

## Explainability & governance
- Rule-based labels give transparent training targets.
- Logistic regression coefficients (inspect via Spark) keep drivers interpretable.
- Agentic layer emits deterministic alert reasons; generative layer provides concise English summaries (no external LLM calls).

## Repository layout
- `src/`: ingestion, features, ML, insights, pipeline.
- `app/`: Flask app, templates, and static assets.
- `outputs/`: generated analytics/ML exports (created after running the pipeline).
- `requirements.txt`: Python dependencies.

## Notes
- If new study folders are added under `QC Anonymized Study Files/`, rerun `python -m src.pipeline`.
- For large datasets, ensure sufficient memory; PySpark Arrow conversion is enabled for speed.
