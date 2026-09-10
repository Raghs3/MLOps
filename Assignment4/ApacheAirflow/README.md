# Workflow Automation Using Apache Airflow

## Objective
Automate a 5-step data pipeline (extract → validate → process → report → notify)
as an Airflow DAG, scheduled daily, and monitor it through the Airflow Web UI.

## Setup
Airflow doesn't officially support running natively on Windows, so this assignment
runs it via Docker (the official `apache/airflow:2.10.4` image + a Postgres 13
metadata database), orchestrated with `docker-compose.yaml`. No local pip install
or venv is required — the image already bundles a supported Python + Airflow.

Services defined in `docker-compose.yaml`:
- **postgres** — Airflow's metadata DB (stores DAG runs, task instances, users).
- **airflow-init** — one-shot container: runs `airflow db migrate` and creates
  the `admin`/`admin` user, then exits.
- **airflow-webserver** — serves the Web UI at `localhost:8080`.
- **airflow-scheduler** — parses DAGs from `dags/` and schedules/executes tasks.

`dags/`, `data/`, `output/`, and `logs/` are bind-mounted into the containers, so
files written by tasks appear directly in this folder on the host.

## How to run
```bash
cd Assignment4/ApacheAirflow
docker compose up -d      # pulls images, starts postgres + init + webserver + scheduler
```
Open `http://localhost:8080` and log in with `admin` / `admin`. In the DAGs list,
unpause `data_pipeline_dag` (toggle switch) and click **Trigger DAG** to run it
manually — it's also scheduled to run automatically once a day (`@daily`).

```bash
docker compose down       # stop everything (metadata DB persists in a named volume)
```

## The DAG — `dags/data_pipeline_dag.py`
Five `PythonOperator` tasks, chained as `extract >> validate >> process >> report >> notify`:

1. **extract** — generates a small sample dataset (20 rows: id, name, age, score,
   some fields intentionally blank/missing) and writes it to `data/raw_data.csv`.
2. **validate** — reads the raw CSV and counts missing `age`/`score` values,
   printing validation statistics.
3. **process** — cleans the data (drops incomplete rows, normalizes name casing,
   casts types) and writes `output/processed_data.csv`.
4. **report** — aggregates counts from the earlier tasks (passed via Airflow's
   **XCom** mechanism, which lets tasks exchange small values) into
   `output/report.json`: records processed, missing values found, clean records
   after processing, and completion status.
5. **notify** — reads the report and prints a success message summarizing the run.

Scheduling: `schedule="@daily"`, `catchup=False` (so it won't backfill missed
runs), `start_date=datetime(2026, 1, 1)`.

## Execution proof
| Screenshot | What it shows |
|---|---|
| `screenshots/dashboard.png` | Airflow dashboard after first login/setup |
| `screenshots/login.png` | Airflow login screen |
| `screenshots/dag_run_success.jpeg` | DAGs list showing `data_pipeline_dag` Active, with a successful run |
| `screenshots/dag_graph_view.jpeg` | Graph view — all 5 tasks green (`success`) in sequence |
| `screenshots/task_logs.png` | Logs for the `report` task, showing the generated report dict printed to stdout |

## Generated deliverable
`output/report.json` from an actual run:
```json
{
  "records_processed": 20,
  "missing_values_identified": 5,
  "clean_records_after_processing": 16,
  "processing_completion_status": "SUCCESS",
  "generated_at": "2026-09-10T11:30:40.576476"
}
```

## Conclusion
This assignment demonstrates workflow orchestration with Apache Airflow: defining
a DAG as a set of Python-callable tasks, wiring linear dependencies with `>>`,
passing intermediate results between tasks via XCom, and observing execution
through the scheduler, Web UI Graph/Grid views, and task logs. Running Airflow in
Docker sidesteps its lack of native Windows support while keeping the setup fully
reproducible via `docker-compose.yaml`.
