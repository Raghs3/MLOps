import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CANDIDATE_NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CANDIDATE_CATEGORICAL = ["Pclass", "Sex", "Embarked"]


def select_features(df, target_col="Survived", k=5):
    """Rank the candidate raw columns by correlation with the target and
    return the names of the top-k to keep.

    Each candidate is imputed/encoded here only to compute a numeric
    correlation score - the real feature engineering pipeline (built
    separately in pipeline.py) re-fits its own imputers/encoders from
    scratch on just the columns this function selects.
    """
    candidates = CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL
    X = df[candidates]
    y = df[target_col]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    scoring_transformer = ColumnTransformer([
        ("num", numeric_pipe, CANDIDATE_NUMERIC),
        ("cat", categorical_pipe, CANDIDATE_CATEGORICAL),
    ])

    encoded = scoring_transformer.fit_transform(X)
    encoded_names = scoring_transformer.get_feature_names_out()

    encoded_df = pd.DataFrame(encoded, columns=encoded_names)
    encoded_df[target_col] = y.values

    correlations = encoded_df.corr(numeric_only=True)[target_col].drop(target_col).abs()

    # Group each encoded column (e.g. "cat__Sex_male") back to the raw
    # column it came from, scoring the raw column by its strongest-
    # correlated encoded value.
    scores = {}
    for encoded_name, corr in correlations.items():
        _, remainder = encoded_name.split("__", 1)
        if remainder in CANDIDATE_NUMERIC:
            raw_name = remainder
        else:
            raw_name = next(c for c in CANDIDATE_CATEGORICAL if remainder.startswith(c))
        scores[raw_name] = max(scores.get(raw_name, 0.0), corr)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = [name for name, _ in ranked[:k]]

    print("\nFeature correlation scores (abs, vs target):")
    for name, score in ranked:
        marker = "*" if name in selected else " "
        print(f"  {marker} {name}: {score:.4f}")

    print("\nSelected Features:")
    for name in selected:
        print(" ", name)

    return selected
