# Monitor a Machine Learning Model using Prometheus and Grafana

An Iris classifier (Random Forest) served with FastAPI, instrumented with `prometheus_client`, scraped by Prometheus, visualised in Grafana and monitored with alert rules routed through Alertmanager. Everything runs locally with Docker Compose.

## Architecture

```
simulate.py -> Model service (FastAPI :8000, /metrics)
                    ^ scrape every 5s
Node Exporter :9100 -> Prometheus :9090 -> Alertmanager :9093
                           ^ PromQL
                       Grafana :3000
```

Prometheus pulls the metrics; Grafana only queries Prometheus and never talks to the service.

## Project Structure

```
Assignment8/ml-monitoring/
├── app.py                 # FastAPI service with Prometheus metrics
├── train.py               # trains the model, saves training statistics
├── simulate.py            # traffic generator (normal | drift | errors)
├── tests/test_api.py      # pytest suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── model/                 # model.joblib + training_stats.json (from train.py)
├── prometheus/            # prometheus.yml, alert_rules.yml
├── alertmanager/          # alertmanager.yml
├── grafana/
│   ├── provisioning/      # datasource + dashboard provider
│   └── dashboards/ml-model-monitoring.json
└── screenshots/
```

## Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `ml_prediction_requests_total` | Counter | endpoint, status | Request volume and rate |
| `ml_prediction_errors_total` | Counter | endpoint | Failed requests (status >= 400) |
| `ml_prediction_latency_seconds` | Histogram | endpoint | Latency, average and p95 |
| `ml_predictions_total` | Counter | predicted_class | Successful predictions per class |
| `ml_prediction_confidence` | Histogram | none | Top-class probability |
| `ml_feature_drift_zscore` | Gauge | feature | Rolling mean of last 100 inputs vs training mean |
| `ml_feedback_total` | Counter | outcome | Ground-truth feedback, used for accuracy |
| `ml_model_info` | Gauge | version | Serving model version |

The first four are the required metrics; the rest add drift and accuracy monitoring. Labels use small fixed sets to avoid high cardinality. Node Exporter adds host CPU and memory metrics.

## Run

```bash
pip install -r requirements.txt
python train.py                      # creates model/ (needed by the Docker build)
pytest -v                            # optional: run the tests
docker compose up -d --build
python simulate.py normal            # keep running in a second terminal
```

Set `GRAFANA_PASSWORD` in the environment to override the default Grafana admin password.

| Service | URL |
|---------|-----|
| Model API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 (check `/targets` and `/alerts`) |
| Alertmanager | http://localhost:9093 |
| Grafana | http://localhost:3000 (dashboard "ML Model Monitoring") |

Sample request:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# {"prediction":"setosa","confidence":1.0,"model_version":"1.0.0"}
```

## Dashboard

Provisioned automatically, grouped in three rows:

- **Service:** total requests, request rate, average latency, p95 latency, error rate, availability (`up{job="ml_api"}`), requests by status
- **Model:** predictions by class, average confidence, feature drift z-score (threshold at +-3), rolling accuracy
- **Infrastructure:** host CPU and memory usage

## Alert Rules

| Alert | Condition | For | Severity |
|-------|-----------|-----|----------|
| ModelServiceDown | `up{job="ml_api"} == 0` | 30s | critical |
| HighErrorRate | error ratio > 5% | 1m | warning |
| HighP95Latency | p95 > 0.5s | 2m | warning |
| FeatureDriftDetected | abs(z-score) > 3 | 1m | warning |
| LowModelAccuracy | accuracy < 85% over 5m | 2m | critical |

## Test Scenarios

| Scenario | Trigger | Expected | Observed |
|----------|---------|----------|----------|
| A. Normal traffic | `python simulate.py normal` | z-scores within +-3, accuracy about 95%, no alerts | _to fill after running_ |
| B. Data drift | `python simulate.py drift` | petal/sepal length z-score above 3, FeatureDriftDetected fires | _to fill_ |
| C. API errors | `python simulate.py errors` | about 30% 422 responses, HighErrorRate fires | _to fill_ |
| D. Outage | `docker compose stop model-service` | status DOWN, ModelServiceDown fires after 30s | _to fill_ |
| E. Recovery | `docker compose start model-service`, then scenario A | alerts resolve, counters restart from zero | _to fill_ |

## Notes

- The service runs a single Uvicorn worker (metrics are kept in process memory).
- The drift check compares only feature means, so it misses changes in shape or correlation.
- Image versions are pinned in `docker-compose.yml`.

## Tech Stack

- Python 3.11, scikit-learn, FastAPI, prometheus_client
- Prometheus, Alertmanager, Node Exporter, Grafana
- Docker Compose, pytest
