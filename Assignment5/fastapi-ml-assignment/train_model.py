from pathlib import Path 
import joblib 
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
 
# Load the Iris dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 
 
# Divide the dataset 
X_train, X_test, y_train, y_test = train_test_split( 
    X, 
    y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y 
) 
 
# Train the classification model 
model = RandomForestClassifier( 
    n_estimators=100, 
    random_state=42 
) 
 
model.fit(X_train, y_train) 
 
# Evaluate the model 
predictions = model.predict(X_test) 
test_accuracy = accuracy_score(y_test, predictions) 
 
# Store the model and related metadata 
model_bundle = { 
    "model": model, 
    "target_names": iris.target_names.tolist(), 
    "feature_names": iris.feature_names, 
    "test_accuracy": float(test_accuracy), 
    "model_version": "1.0.0" 
} 
 
# Create the artifacts directory
artifact_directory = Path(__file__).resolve().parent / "artifacts"
artifact_directory.mkdir(exist_ok=True)
 
# Save the model 
model_path = artifact_directory / "iris_model.joblib" 
joblib.dump(model_bundle, model_path) 
 
print(f"Model saved at: {model_path}") 
print(f"Test accuracy: {test_accuracy:.4f}")
