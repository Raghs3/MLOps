from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "iris_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once when the application starts.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found. Run train_model.py first."
        )

    app.state.model_bundle = joblib.load(MODEL_PATH)

    yield

    # Release the model during application shutdown
    app.state.model_bundle = None


app = FastAPI(
    title="Iris Classification API",
    description="REST API for Iris flower classification",
    version="1.0.0",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):
    sepal_length: float = Field(
        gt=0,
        le=20,
        description="Sepal length in centimetres"
    )

    sepal_width: float = Field(
        gt=0,
        le=20,
        description="Sepal width in centimetres"
    )

    petal_length: float = Field(
        gt=0,
        le=20,
        description="Petal length in centimetres"
    )

    petal_width: float = Field(
        gt=0,
        le=20,
        description="Petal width in centimetres"
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_species: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


@app.get("/")
def root():
    """
    Display basic API information.
    """

    return {
        "message": "Welcome to the Iris Classification API",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check(request: Request):
    """
    Check whether the API and model are available.
    """

    model_loaded = (
        request.app.state.model_bundle is not None
    )

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }


@app.get("/model-info")
def model_information(request: Request):
    """
    Return model metadata.
    """

    bundle = request.app.state.model_bundle

    return {
        "model_type": type(bundle["model"]).__name__,
        "model_version": bundle["model_version"],
        "feature_names": bundle["feature_names"],
        "target_names": bundle["target_names"],
        "test_accuracy": bundle["test_accuracy"]
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    data: PredictionRequest,
    request: Request
):
    """
    Generate an Iris flower prediction.
    """

    bundle = request.app.state.model_bundle
    model = bundle["model"]
    target_names = bundle["target_names"]

    features = np.array(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]
    )

    try:
        predicted_class = int(model.predict(features)[0])
        class_probabilities = model.predict_proba(features)[0]

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The model could not generate a prediction."
        ) from error

    probability_dictionary = {
        species: round(float(probability), 4)
        for species, probability in zip(
            target_names,
            class_probabilities
        )
    }

    confidence = float(
        class_probabilities[predicted_class]
    )

    return {
        "predicted_class": predicted_class,
        "predicted_species": target_names[predicted_class],
        "confidence": round(confidence, 4),
        "probabilities": probability_dictionary,
        "model_version": bundle["model_version"]
    }
