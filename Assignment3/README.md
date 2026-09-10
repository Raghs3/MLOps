# Titanic — Feature Engineering Pipeline

## Dataset
Classic Titanic dataset (891 rows, 12 columns), fetched via pandas and saved to `data/titanic.csv`.

## Step 1: Load Dataset
`src/preprocessing.py` reads the CSV, prints shape, `.info()`, and dtypes to separate numeric from categorical columns.

## Step 2: Data Cleaning
- Checked and reported missing values per column, and duplicate rows (0 found).
- Dropped `PassengerId`, `Name`, `Ticket` — identifiers/free text with no learnable signal for a tabular model.
- Dropped `Cabin` — ~77% missing, imputing would mean fabricating most of the column.
- Remaining missing values (`Age`, `Embarked`) are handled by `SimpleImputer` *inside* the pipeline (Step 3/5) rather than manually beforehand, so the saved pipeline can also impute missing values on brand-new inference data, not just the training set.

## Step 3 & 4: Feature Engineering + Feature Selection
`src/feature_selection.py` scores 7 candidate columns (`Age`, `SibSp`, `Parch`, `Fare` as numeric; `Pclass`, `Sex`, `Embarked` as categorical) by correlation with `Survived`, after imputing/encoding them purely for scoring purposes. The top 5 by absolute correlation are kept.

- **Numerical**: median imputation + `StandardScaler`
- **Categorical**: most-frequent imputation + `OneHotEncoder`
- **Selection method**: correlation analysis

**Selected features:** `Sex`, `Pclass`, `Fare`, `Embarked`, `Parch`

(`Age` was edged out of the top 5 by `Parch` — 0.0649 vs 0.0816 absolute correlation with `Survived`. This differs slightly from the PDF's illustrative example, which is expected: it reflects the real statistics of this dataset rather than a scripted answer.)

## Step 5: Build Pipeline
`src/pipeline.py` builds per-type `Pipeline`s (impute → scale / impute → encode) combined via `ColumnTransformer`, wrapped in one outer `Pipeline`.

## Step 6: Transform Dataset
- **Original Dataset Shape (before selection):** (891, 8) — after dropping `PassengerId`/`Name`/`Ticket`/`Cabin`
- **Transformed Dataset Shape:** (891, 10) — matches the PDF's example shape exactly

## Step 7: Save Pipeline
Saved with `joblib` to `models/feature_pipeline.pkl`.

## Step 8: Reuse Pipeline
`pipeline.py` loads the saved pipeline and transforms a hand-built new sample row, demonstrating it works on unseen raw data (including re-imputing/re-encoding without retraining).

## How to run
```powershell
pip install -r requirements.txt
python src\pipeline.py
```
This single command runs the entire flow end to end: load → clean → select features → build pipeline → transform → save → reload and transform a new sample.

## Files
- `data/titanic.csv` — raw dataset
- `src/preprocessing.py` — Step 1 (load) + Step 2 (cleaning)
- `src/feature_selection.py` — Step 4 (correlation-based feature selection)
- `src/pipeline.py` — Steps 3, 5, 6, 7, 8 (build, transform, save, reuse) — main entry point
- `models/feature_pipeline.pkl` — saved fitted pipeline
- `output/transformed_data.csv` — transformed dataset
