import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

df.to_csv("./data/iris.csv", index=False)

# print(df.head())
# print(df.shape)

from sklearn.preprocessing import StandardScaler, LabelEncoder

print("Missing values per column:\n", df.isnull().sum())

encoder = LabelEncoder()
df["species_encoded"] = encoder.fit_transform(df["species"])

feature_cols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

X = pd.DataFrame(X_scaled, columns=feature_cols)
y = df["species_encoded"]

print(X.head())
print(y.head())