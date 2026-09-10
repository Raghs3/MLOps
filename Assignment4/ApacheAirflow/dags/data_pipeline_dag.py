import csv
import json
import os
import random
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = "/opt/airflow/data"
OUTPUT_DIR = "/opt/airflow/output"
RAW_PATH = os.path.join(DATA_DIR, "raw_data.csv")
PROCESSED_PATH = os.path.join(OUTPUT_DIR, "processed_data.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.json")

COLUMNS = ["id", "name", "age", "score"]


def extract(**context):
    os.makedirs(DATA_DIR, exist_ok=True)

    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi"]
    rows = []
    for i in range(1, 21):
        row = {
            "id": i,
            "name": random.choice(names),
            "age": random.choice([18, 21, 25, 30, None]),  # None -> missing value
            "score": round(random.uniform(40, 100), 1) if random.random() > 0.1 else "",
        }
        rows.append(row)

    with open(RAW_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Data Extracted: {len(rows)} records saved to {RAW_PATH}")
    context["ti"].xcom_push(key="record_count", value=len(rows))


def validate(**context):
    with open(RAW_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    missing_age = sum(1 for r in rows if r["age"] in ("", "None", None))
    missing_score = sum(1 for r in rows if r["score"] in ("", "None", None))
    total_missing = missing_age + missing_score

    print(
        f"Data Validated: {len(rows)} records checked, "
        f"missing age={missing_age}, missing score={missing_score}"
    )
    context["ti"].xcom_push(key="missing_values", value=total_missing)


def process(**context):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(RAW_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cleaned = []
    for r in rows:
        if not r["age"] or not r["score"]:
            continue  # drop incomplete records
        cleaned.append(
            {
                "id": r["id"],
                "name": r["name"].strip().title(),
                "age": int(r["age"]),
                "score": float(r["score"]),
            }
        )

    with open(PROCESSED_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(cleaned)

    print(f"Data Processed: {len(cleaned)} clean records saved to {PROCESSED_PATH}")
    context["ti"].xcom_push(key="processed_count", value=len(cleaned))


def report(**context):
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="extract", key="record_count")
    missing_values = ti.xcom_pull(task_ids="validate", key="missing_values")
    processed_count = ti.xcom_pull(task_ids="process", key="processed_count")

    summary = {
        "records_processed": record_count,
        "missing_values_identified": missing_values,
        "clean_records_after_processing": processed_count,
        "processing_completion_status": "SUCCESS",
        "generated_at": datetime.utcnow().isoformat(),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Report Generated: {summary}")


def notify(**context):
    with open(REPORT_PATH) as f:
        summary = json.load(f)
    print(
        "Notification Sent: data_pipeline_dag completed successfully — "
        f"{summary['clean_records_after_processing']} clean records "
        f"out of {summary['records_processed']} processed."
    )


with DAG(
    dag_id="data_pipeline_dag",
    description="Extract, validate, process, report and notify — a daily data pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["assignment4", "mlops"],
) as dag:

    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="validate", python_callable=validate)
    t3 = PythonOperator(task_id="process", python_callable=process)
    t4 = PythonOperator(task_id="report", python_callable=report)
    t5 = PythonOperator(task_id="notify", python_callable=notify)

    t1 >> t2 >> t3 >> t4 >> t5
