# Smart Parking Intelligence for Indian Urban Areas

This project implements an end-to-end smart parking analytics and forecasting system using the `parkingStream 2.csv` dataset. The pipeline cleans the raw parking data, engineers time-series features, trains forecasting models, generates evaluation artifacts, writes the data to SQLite for SQL-based analysis, and exposes the results through a Streamlit dashboard.

## Core Highlights

- Indian parking dataset with 14 parking systems and time-series utilization records.
- Data-quality pipeline that fixes occupancy-capacity inconsistencies before modeling.
- Spatial-temporal feature engineering with nearest-neighbor parking pressure and previous-day same-slot memory.
- Multi-horizon forecasting for 30 minutes, 1 hour, and 2 hours ahead.
- Calibrated uncertainty intervals for advanced forecasts.
- Recommendation engine that ranks parking systems by availability, congestion, and risk.
- Anomaly detection for unusual parking-demand patterns.
- One-hour-ahead parking utilization forecasting using:
  - Persistence baseline
  - Random Forest
  - XGBoost
  - LSTM
  - Weighted Ensemble
- Syllabus-aligned components:
  - Data science methodology and lifecycle
  - Data preprocessing and EDA
  - Time-series analytics
  - SQL/SQLite querying
  - Clustering and dimensionality reduction using KMeans + PCA
  - Dashboard-based visualization

## Project Structure

- `data/raw/parkingStream_2.csv`: local source dataset
- `data/processed/`: cleaned data and model-ready features
- `artifacts/models/`: trained models
- `artifacts/plots/`: generated charts
- `artifacts/reports/`: metrics, forecasts, and project summaries
- `artifacts/db/smart_parking.db`: SQLite analytics database
- `src/smart_parking/`: reusable project modules
- `scripts/run_pipeline.py`: main pipeline runner
- `dashboard/app.py`: Streamlit dashboard
- `docs/`: course-facing documentation

## Setup

Use the project virtual environment:

```bash
brew install libomp
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run The Pipeline

```bash
.venv/bin/python scripts/run_pipeline.py
```

## Launch The Dashboard

```bash
.venv/bin/streamlit run dashboard/app.py
```

## Run The API Locally

```bash
.venv/bin/python scripts/run_api.py
```

Swagger docs will then be available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

## Deploy The API

The repository includes a `Dockerfile` and `render.yaml` for hosting the FastAPI backend as a separate web service. A simple way to publish the API is:

1. Push the repository to GitHub.
2. Create a new Render web service from the repository.
3. Use the root `Dockerfile`.
4. After deployment, open:
   - `https://<your-service>.onrender.com/docs`
   - `https://<your-service>.onrender.com/openapi.json`

This makes the API accessible without running it from a local laptop and allows external software to consume the forecasts, recommendations, anomalies, and monitoring endpoints directly.

## Main Prediction Target

The project forecasts `target_utilization_1h`, which represents the parking lot utilization ratio one hour after the current observation. This is more informative than binary occupied/empty prediction because it reflects real parking pressure across large-capacity parking locations.

## Advanced Outputs

- `artifacts/reports/multi_horizon_metrics.csv`: forecast performance across multiple horizons
- `artifacts/reports/parking_recommendations.csv`: ranked parking options with risk-aware scores
- `artifacts/reports/demand_anomalies.csv`: detected abnormal parking-demand events
- `data/processed/spatial_neighbor_graph.csv`: nearest-neighbor graph used for spatial context
