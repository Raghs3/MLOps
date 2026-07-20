import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

df.to_csv("./data/iris.csv", index=False)
print(df.head())
print(df.shape)
