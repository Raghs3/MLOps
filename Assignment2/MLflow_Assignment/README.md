# Iris Classification — MLflow Experiment Tracking

## Dataset
Iris dataset (150 samples, 4 features, 3 species) via scikit-learn.

## Preprocessing
- Checked for missing values (none found)
- Label-encoded the `species` column (setosa/versicolor/virginica → 0/1/2)
- Standard-scaled the 4 numeric features

## Split
80% train / 20% test, stratified by class.

## Models trained
- Logistic Regression
- Decision Tree
- Random Forest

## Results
All three models achieved identical metrics on the test set:
accuracy = precision = recall = f1_score = 0.933

## Model selection
Since all three models tied exactly on this test set, Random Forest was 
selected because ensembling multiple trees generally makes it more robust 
to noise and overfitting than a single Decision Tree, and it's competitive 
with Logistic Regression while requiring no assumption of linear separability.

## Registered model
`IrisBestModel`, Version 1, registered in MLflow Model Registry from the
Random Forest run.

## How to run
1. `pip install -r requirements.txt`
2. `python src/train.py`
3. `mlflow ui --backend-store-uri sqlite:///mlflow.db`
