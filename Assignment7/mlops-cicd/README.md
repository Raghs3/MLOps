# MLOps CI/CD Pipeline with GitHub Actions

A simple machine learning project (Logistic Regression on the Iris dataset) with an automated CI/CD pipeline built using GitHub Actions.

## Pipeline Overview

```
Code Commit -> Build -> Test -> Deploy
```

The workflow runs on every push or pull request to the `main` branch:

1. **Test job (CI):** checks out the code, sets up Python 3.11, installs dependencies and runs `pytest`.
2. **Deploy job (CD):** runs only if the tests pass and the code is on `main`. Deployment is simulated with `echo`.

If the tests fail, the pipeline stops and deployment is skipped.

## Project Structure

```
Assingment7/mlops-cicd/
├── src/model.py                  # Model training code
├── tests/test_model.py           # Unit tests
├── requirements.txt              # Dependencies
├── app.py
└── README.md

.github/workflows/ci-cd.yml       # GitHub Actions workflow (repo root)
```

The workflow lives at the repository root because GitHub Actions only reads `.github/workflows/` there. It runs inside `Assingment7/mlops-cicd` and only triggers when files in that folder change.

## Run Locally

```bash
pip install -r requirements.txt
python src/model.py      # prints the model accuracy
pytest -v                # runs the unit tests
```

## Test Rule

`tests/test_model.py` asserts that model accuracy is at least `0.80`. Changing the threshold to `0.99` makes the test fail, which demonstrates that deployment is blocked when validation fails.

## Viewing Pipeline Results

Open the repository on GitHub and go to the **Actions** tab, then select **MLOps CI/CD Pipeline**.

| Scenario | Test job | Deploy job |
|----------|----------|------------|
| Tests pass on `main` | Passed | Runs |
| Tests fail | Failed | Skipped |
| Pull request | Runs | Skipped |

## Tech Stack

- Python 3.11
- scikit-learn
- pytest
- GitHub Actions
