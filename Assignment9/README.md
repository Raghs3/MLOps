# Assignment 9: Train and Deploy a Machine Learning Model using AWS SageMaker

MLOps (AI3206C) | Vishwakarma Institute of Technology, Pune

A Random Forest classifier is trained on AWS SageMaker, deployed as a real-time endpoint, and exposed through AWS Lambda and Amazon API Gateway (`POST /predict`).

## Problem
Predict whether a breast tumour is **malignant (0)** or **benign (1)** from 30 cell-nucleus measurements.
Dataset: Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn (569 samples, 212 malignant, 357 benign). Split: 80% train / 20% test, stratified, `random_state=42`.

## Files
| File | Purpose |
|---|---|
| `breast_cancer_sagemaker.ipynb` | Notebook: data preparation, S3 upload, two training jobs, deployment, endpoint test and confusion matrix |
| `train.py` | SageMaker script-mode training script (Random Forest, prints metrics, saves `model.joblib`) |
| `lambda_function.py` | Lambda handler: validates 30 features, calls the endpoint, returns `benign` or `malignant` |

## Architecture
```
Client -> API Gateway (POST /predict, API key + throttling)
       -> Lambda (validate input, InvokeEndpoint)
       -> SageMaker real-time endpoint (model.tar.gz from S3)

Notebook -> S3 (train.csv, test.csv) -> SageMaker training job (scikit-learn container from ECR)
         -> model.tar.gz in S3
CloudWatch: training logs, endpoint metrics (Invocations, ModelLatency, OverheadLatency, 4XX/5XX), 5XX alarm
```

## Environment
- Region: `ap-south-1` (Mumbai)
- SageMaker Python SDK v2 (`pip install "sagemaker<3"`), scikit-learn container `1.2-1`
- Training instance: `ml.m5.large`; endpoint instance: `ml.t2.medium`
- Lambda runtime: Python 3.12, environment variable `ENDPOINT_NAME=breast-cancer-rf-endpoint`

## Results (test set, 114 samples)
| Metric | Run 1 (depth 8, 200 trees) | Run 2 (depth 4, 100 trees) |
|---|---|---|
| Accuracy | 0.9561 | 0.9561 |
| Precision (benign) | 0.9589 | 0.9589 |
| Recall (benign) | 0.9722 | 0.9722 |
| F1-score | 0.9655 | 0.9655 |
| ROC-AUC | 0.9940 | 0.9950 |
| Recall (malignant) | 0.9286 | 0.9286 |
| Confusion matrix (TN / FP / FN / TP) | 39 / 3 / 2 / 70 | 39 / 3 / 2 / 70 |
| Training / billable seconds | 124 / 124 | 134 / 134 |

Run 2 (the simpler model) was deployed. The 3 false-benign cases (malignant tumours predicted as benign) are the costly error in a medical setting, so malignant-class recall is the metric to prioritise.

## How to run
1. Create an IAM execution role for SageMaker (`AmazonSageMakerFullAccess` plus a policy limited to your bucket) and a private, encrypted S3 bucket in the same region.
2. In a SageMaker notebook instance, open `breast_cancer_sagemaker.ipynb`, set `bucket`, and set `role = sagemaker.get_execution_role()`.
3. Run the notebook cells: upload data, train both estimators, then `deploy()` the better one with endpoint name `breast-cancer-rf-endpoint`.
4. Create a Lambda function from `lambda_function.py` with the `ENDPOINT_NAME` environment variable, a 30 s timeout, and an IAM policy that allows only `sagemaker:InvokeEndpoint` on this endpoint.
5. Create a REST API in API Gateway: `POST /predict` with Lambda proxy integration, API key required, a usage plan with throttling, deployed to stage `prod`.
6. Test:
   ```bash
   curl -X POST "$API_URL" -H "x-api-key: $API_KEY" -H "Content-Type: application/json" -d @sample.json
   ```
   where `sample.json` is `{"features": [ ...30 numbers... ]}`.
7. **Clean up** when done: delete the endpoint, the API, the Lambda function, and stop the notebook instance.

## Notes and fixes
- `train.py` reads `SM_MODEL_DIR`, `SM_CHANNEL_TRAIN` and `SM_CHANNEL_TEST`, so the same script runs locally and on SageMaker. It also reports malignant-class recall and the confusion-matrix counts.
- Sending a single sample to the endpoint as CSV made the container read a 1-D array and return HTTP 500. The Lambda now sends the sample as a 2-D JSON array (`[[...30 values...]]`).
- No credentials, API keys or account IDs are stored in this repository. Keys were used only as terminal environment variables.