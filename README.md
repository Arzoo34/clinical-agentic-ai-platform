# Agentic AI Platform for Clinical Trial Data Intelligence

This repository contains a multi-agent, event-driven pipeline that ingests heterogeneous clinical trial data, harmonizes schemas, monitors data quality and operational performance, detects risks, and generates explainable insights with automated follow-up actions. A Streamlit UI is provided for interactive runs.

## Quick Start

Prerequisites: Python 3.10+, recommended inside a virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Run the test suite:

```bash
pytest
```

Run the interactive Streamlit UI:

```bash
streamlit run streamlit_app.py
```

Demo data is located in `data/demo` (and mirrored via junction to `Data for problem Statement 1/QC Anonymized Study Files`). You can also upload a custom ZIP of Excel files directly in the UI.

## Project Layout

- `app/agents/`: One file per agent (ingestion, schema harmonization, quality, operational, anomaly, risk scoring, insights, alerts, base).
- `app/core/`: Event bus, config loader, logging, models, and storage utilities.
- `config/config.yaml`: Thresholds, required columns, mappings, and paths.
- `data/`: Local filesystem storage (`raw/`, `staging/`, `curated/`, `logs/`, `demo/`).
- `streamlit_app.py`: Streamlit UI and orchestrated demo dashboard.
- `tests/test_platform.py`: Comprehensive automated test suite.

## Pipeline (Orchestrated)

1. **File Ingestion Agent** – Accepts ZIP uploads, extracts XLSX files, captures metadata, validates basic structure, and writes raw copies to `data/raw`.
2. **Schema Mapping & Harmonization Agent** – Normalizes column names, applies alias mapping, and emits harmonized datasets into `data/curated`.
3. **Data Quality Monitoring Agent** – Computes missingness, duplicate detection, simple outlier detection, protocol deviation checks, and quality scores.
4. **Operational Intelligence Agent** – Tracks submission freshness, file-level row counts, and simple site performance KPIs.
5. **Anomaly Detection Agent** – Flags statistical anomalies (row-count z-scores, sudden shifts).
6. **Risk Scoring & Prioritization Agent** – Combines quality, operational, and anomaly signals into composite risk scores.
7. **Generative Insight & Explanation Agent** – Creates human-readable summaries that explain why alerts or risks were triggered (LLM pluggable).
8. **Task Automation & Alert Agent** – Emits alerts, suggested tasks, and escalation markers.

## Running on Provided Dataset

The extracted dataset lives at `Data for problem Statement 1/QC Anonymized Study Files`, containing subfolders `Study 1...` through `Study 25...` with multiple Excel files each.

- From the Streamlit UI, choose **Use demo folder** to process those files.
- Or upload a new ZIP; the ingestion agent will extract and process it end-to-end.

## Outputs

- Harmonized parquet/CSV files under `data/curated/`.
- Data quality and risk reports under `data/logs/`.
- Operational KPI and anomaly summaries displayed in Streamlit.
- Human-readable insights and alert/task logs displayed in Streamlit and written to `data/logs/alerts.jsonl`.

## Configuration

Edit `config/config.yaml` to adjust:
- Column alias mappings
- Required fields
- Thresholds for missingness, anomaly z-scores, and risk scoring weights
- Default data roots for demo runs

## Notes

- The orchestrator is intentionally lightweight and file-based to keep the demo reproducible.
- LLM usage is abstracted behind a simple function; by default it uses templated text for offline use. Swap in your OpenAI-compatible client if desired.***

