import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

from src.config import load_config, resolve_path
from src.predict import PredictionError, make_prediction


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Configuration
# ============================================================

config = load_config()

model_config = config["model"]

MODEL_VERSION = model_config["version"]

MODEL_PATH = resolve_path(
    config,
    model_config["path"],
)

PREPROCESSOR_PATH = resolve_path(
    config,
    model_config["preprocessor_path"],
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=getattr(
        logging,
        config["logging"]["level"],
        logging.INFO,
    ),
    format=config["logging"]["format"],
)

logger = logging.getLogger(__name__)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Olist Late Delivery Prediction API",
    description=(
        "Machine learning API for predicting whether "
        "an Olist order will be delivered late."
    ),
    version=MODEL_VERSION,
)


# ============================================================
# Global model objects
# ============================================================

model = None
preprocessor = None
required_features = []


# ============================================================
# Load artifacts
# ============================================================

def load_model_artifacts():
    """
    Load the trained model and fitted preprocessor.

    No training or fitting happens here.
    """

    global model
    global preprocessor
    global required_features

    logger.info(
        "Loading model from: %s",
        MODEL_PATH,
    )

    logger.info(
        "Loading preprocessor from: %s",
        PREPROCESSOR_PATH,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found: "
            f"{PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    # Get the exact columns expected by the saved
    # ColumnTransformer.
    if hasattr(preprocessor, "feature_names_in_"):
        required_features = list(
            preprocessor.feature_names_in_
        )

    elif hasattr(preprocessor, "transformers_"):
        required_features = []

        for _, transformer, columns in preprocessor.transformers_:
            if columns == "drop":
                continue

            if columns == "passthrough":
                continue

            required_features.extend(
                list(columns)
            )

    else:
        raise RuntimeError(
            "Could not determine required features "
            "from the saved preprocessor."
        )

    logger.info(
        "Model loaded successfully."
    )

    logger.info(
        "Preprocessor loaded successfully."
    )

    logger.info(
        "Number of required features: %d",
        len(required_features),
    )


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Load ML artifacts when FastAPI starts.
    """

    try:
        load_model_artifacts()

    except Exception:
        logger.exception(
            "Failed to load model artifacts."
        )

        raise


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Olist Late Delivery Prediction API",
        "version": MODEL_VERSION,
        "status": "running",
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():

    return HealthResponse(
        status=(
            "healthy"
            if model is not None
            and preprocessor is not None
            else "unhealthy"
        ),
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        model_version=MODEL_VERSION,
    )


# ============================================================
# Model information endpoint
# ============================================================

@app.get("/model/info")
def model_info():

    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "model_type": type(model).__name__,
        "model_version": MODEL_VERSION,
        "number_of_required_features": len(
            required_features
        ),
        "required_features": required_features,
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:

        # Convert Pydantic request to dictionary.
        input_data = request.model_dump()

        # Convert to DataFrame.
        df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # Important:
        #
        # We only use the features expected by the saved
        # preprocessor.
        # ----------------------------------------------------

        result = make_prediction(
            data=df,
            model=model,
            preprocessor=preprocessor,
            model_version=MODEL_VERSION,
            required_features=required_features,
        )

        return result

    except PredictionError as exc:

        logger.error(
            "Prediction error: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected prediction error."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from exc
