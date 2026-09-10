# Iris Classification REST API (FastAPI)

## Aim

Design, develop, test, and document a REST API using FastAPI that serves predictions
from a trained Iris flower classification model.

## Project structure

```
fastapi-ml-assignment/
├── artifacts/
│   └── iris_model.joblib
├── tests/
│   └── test_api.py
├── main.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup

This assignment uses the shared virtual environment at the repository root
(`MLOps/.venv`).

```powershell
# from the repository root
.venv\Scripts\activate
pip install -r Assignment5/fastapi-ml-assignment/requirements.txt
```

## Train the model

```powershell
cd Assignment5/fastapi-ml-assignment
python train_model.py
```

Expected output:

```
Model saved at: ...\artifacts\iris_model.joblib
Test accuracy: 0.9000
```

This trains a `RandomForestClassifier` on the Iris dataset (80/20 stratified split,
`random_state=42`) and saves the model together with its metadata
(`target_names`, `feature_names`, `test_accuracy`, `model_version`) to
`artifacts/iris_model.joblib`.

## Run the API

```powershell
cd Assignment5/fastapi-ml-assignment
uvicorn main:app --reload
```

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000/` | Root endpoint |
| `http://127.0.0.1:8000/health` | Health check |
| `http://127.0.0.1:8000/model-info` | Model information |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI schema |

## Example request/response

**Request** — `POST /predict`

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response**

```json
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "model_version": "1.0.0"
}
```

Each input field (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) must
be a number greater than 0 and less than or equal to 20. Invalid input (negative
values, missing fields, or wrong data types) is rejected automatically with a
`422 Unprocessable Content` response.

## Test using cURL

```bash
curl http://127.0.0.1:8000/health

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Run the automated tests

```powershell
cd Assignment5/fastapi-ml-assignment
pytest -v
```

`tests/test_api.py` covers all required cases:

| Test | Input condition | Expected status |
|---|---|---|
| Root endpoint | `GET /` | 200 |
| Health check | Model loaded | 200 |
| Model information | `GET /model-info` | 200 |
| Valid prediction | All values valid | 200 |
| Negative value | One feature below zero | 422 |
| Missing feature | `petal_width` omitted | 422 |
| Incorrect data type | Text instead of a number | 422 |
| Unknown endpoint | `GET /unknown` | 404 |

## HTTP status codes to observe

| Status code | Meaning |
|---|---|
| 200 | Request successfully processed |
| 404 | Endpoint not found |
| 422 | Request validation failed |
| 500 | Internal prediction failure |
| 503 | Model or service unavailable |

## Containerization and deployment (AWS)

The API can be containerized by writing a `Dockerfile` that installs
`requirements.txt` and runs `uvicorn main:app --host 0.0.0.0 --port 8000`, then
built into an image and pushed to Amazon ECR. From there it can be deployed on AWS
via ECS/Fargate (or EC2) behind an Application Load Balancer, or packaged for AWS
Lambda using an ASGI adapter such as Mangum for a serverless deployment.
