# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save trained model to disk
joblib.dump(model, 'iris_model.pkl')
print('Model trained and saved as iris_model.pkl')
