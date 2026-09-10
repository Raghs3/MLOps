# Workflow Automation Using Apache Airflow

## Objective
Automate a 5-step data pipeline (extract → validate → process → report → notify)
as an Airflow DAG, scheduled daily, and monitor it through the Airflow Web UI.

## Setup
Airflow doesn't officially support running natively on Windows, so this assignment
runs it via Docker (the official `apache/airflow` image + Postgres metadata DB),
orchestrated with `docker-compose.yaml`. No local pip install or venv is required.

_(Full run instructions, DAG explanation, and screenshots are added in later steps.)_
