import pandas as pd
from sklearn.datasets import load_iris
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Classification_Experiments")

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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

models = {
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print(f"{name}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

        # Confusion matrix as an image art
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig("confusion_matrix.png", bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # Classification report as a text artifact
        report = classification_report(y_test, preds, target_names=encoder.classes_)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
