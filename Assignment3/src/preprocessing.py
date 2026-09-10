import pandas as pd

DATA_PATH = "data/titanic.csv"
SOURCE_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def load_dataset():
    df = pd.read_csv(SOURCE_URL)
    df.to_csv(DATA_PATH, index=False)

    print("Dataset Loaded Successfully")
    print("Original Dataset Shape:", df.shape)
    print(df.info())
    print("\nFeature types:\n", df.dtypes)
    return df


def clean_dataset(df):
    print("\nMissing values per column:\n", df.isnull().sum())

    duplicates = df.duplicated().sum()
    print("\nDuplicate rows found:", duplicates)
    df = df.drop_duplicates()

    # PassengerId is just a row index, Name and Ticket are near-unique free
    # text/identifier strings - none carry learnable signal for a tabular
    # model, so they're dropped rather than encoded.
    df = df.drop(columns=["PassengerId", "Name", "Ticket"])

    # Cabin is missing for ~77% of passengers. Imputing it would mean
    # fabricating a cabin for most rows, so the column is dropped instead -
    # the remaining missing values (Age, Embarked) are handled later by the
    # SimpleImputer steps inside the actual feature engineering pipeline,
    # which is what makes the saved pipeline reusable on new raw data too.
    df = df.drop(columns=["Cabin"])

    print("\nColumns after dropping identifiers/high-missingness columns:", list(df.columns))
    return df


if __name__ == "__main__":
    raw_df = load_dataset()
    cleaned_df = clean_dataset(raw_df)
    print("\nRemaining missing values (imputed later, inside the pipeline):")
    print(cleaned_df.isnull().sum())
