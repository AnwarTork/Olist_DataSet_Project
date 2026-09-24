import pytest
import joblib

from src.config import (
    load_config,
    resolve_path,
)


def test_saved_model_has_prediction_interface():
    config = load_config()

    model_path = resolve_path(
        config,
        config["model"]["path"],
    )

    if not model_path.exists():
        pytest.skip(
            "Trained model artifact is not present."
        )

    model = joblib.load(
        model_path
    )

    assert callable(
        model.predict
    )

    assert hasattr(
        model,
        "predict_proba",
    )