
import logging
import time
from typing import List

import pandas as pd


logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when prediction fails."""


def validate_input(data: pd.DataFrame) -> None:
    """
    Validate prediction input.
    """

    if not isinstance(data, pd.DataFrame):
        raise PredictionError("Input must be a pandas DataFrame.")

    if data.empty:
        raise PredictionError("Input DataFrame is empty.")


def validate_required_features(
    data: pd.DataFrame,
    required_features: List[str],
) -> None:
    """
    Check that all required features exist.
    """

    missing_features = [
        feature
        for feature in required_features
        if feature not in data.columns
    ]

    if missing_features:
        raise PredictionError(
            f"Missing required features: {missing_features}"
        )


def prepare_input(
    data: pd.DataFrame,
    required_features: List[str],
) -> pd.DataFrame:
    """
    Prepare input DataFrame for inference.

    Important:
    No fitting is performed here.
    """

    validate_input(data)

    validate_required_features(
        data,
        required_features,
    )

    return data[required_features].copy()


def make_prediction(
    data: pd.DataFrame,
    model,
    preprocessor,
    model_version: str,
    required_features: List[str],
):
    """
    Make prediction using saved model and saved preprocessor.

    The preprocessor is only transformed.
    It is never fitted during inference.
    """

    start_time = time.perf_counter()

    try:
        prepared_data = prepare_input(
            data=data,
            required_features=required_features,
        )

        logger.info(
            "Prediction request received with shape=%s",
            prepared_data.shape,
        )

        # IMPORTANT:
        # Do NOT call fit() or fit_transform() here.
        X = preprocessor.transform(prepared_data)

        prediction = model.predict(X)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)

            probability_on_time = float(probabilities[0][0])
            probability_late = float(probabilities[0][1])
        else:
            probability_late = float(prediction[0])
            probability_on_time = 1.0 - probability_late

        prediction_value = int(prediction[0])

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Prediction completed in %.4f seconds",
            elapsed,
        )

        return {
            "prediction": prediction_value,
            "prediction_label": (
                "late"
                if prediction_value == 1
                else "on_time"
            ),
            "probability_late": probability_late,
            "probability_on_time": probability_on_time,
            "model_version": model_version,
        }

    except PredictionError:
        raise

    except Exception as exc:
        logger.exception(
            "Prediction failed."
        )

        raise PredictionError(
            f"Prediction failed: {exc}"
        ) from exc