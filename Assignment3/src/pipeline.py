import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import load_dataset, clean_dataset
from feature_selection import select_features, CANDIDATE_NUMERIC, CANDIDATE_CATEGORICAL

MODEL_PATH = "models/feature_pipeline.pkl"
OUTPUT_PATH = "output/transformed_data.csv"
TARGET_COL = "Survived"


def build_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    # Wrapped in an outer Pipeline (even though it's a single step) so the
    # saved object demonstrates both Pipeline and ColumnTransformer, and so
    # more steps (e.g. a model) could be appended later without changing
    # how this object is saved/loaded/called.
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])
    return full_pipeline


def demonstrate_reuse(selected_features):
    print("\n--- Step 8: Reusing the saved pipeline on a new sample ---")
    loaded_pipeline = joblib.load(MODEL_PATH)

    # One representative value per candidate column so this works no matter
    # which 5 columns select_features() actually picked.
    sample_values = {
        "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 15.50,
        "Pclass": 2, "Sex": "female", "Embarked": "S",
    }
    new_sample = pd.DataFrame([sample_values])[selected_features]

    transformed_sample = loaded_pipeline.transform(new_sample)
    print("New sample (raw):\n", new_sample)
    print("New sample transformed shape:", transformed_sample.shape)
    print("New sample transformed values:\n", transformed_sample)


def main():
    df = load_dataset()
    df = clean_dataset(df)

    selected_features = select_features(df, target_col=TARGET_COL, k=5)
    numeric_features = [f for f in selected_features if f in CANDIDATE_NUMERIC]
    categorical_features = [f for f in selected_features if f in CANDIDATE_CATEGORICAL]

    X = df[selected_features]
    y = df[TARGET_COL]

    print("\nOriginal Dataset Shape:", df.shape)

    pipeline = build_pipeline(numeric_features, categorical_features)
    X_transformed = pipeline.fit_transform(X)

    assert not np.isnan(X_transformed).any(), "Unexpected NaNs remain after imputation"
    print("Missing Values Removed")
    print("Categorical Features Encoded")
    print("Numerical Features Scaled")
    print("\nTransformed Dataset Shape:", X_transformed.shape)

    feature_names = pipeline.get_feature_names_out()
    transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    transformed_df[TARGET_COL] = y.values
    transformed_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nTransformed dataset saved to {OUTPUT_PATH}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline Saved Successfully -> {MODEL_PATH}")

    demonstrate_reuse(selected_features)


if __name__ == "__main__":
    main()
