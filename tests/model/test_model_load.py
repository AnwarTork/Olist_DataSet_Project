from pathlib import Path

import pytest

from src.config import (
    load_config,
    resolve_path,
)

from src.train import load_model


def test_model_artifact_can_be_loaded():
    config = load_config()

    model_path = resolve_path(
        config,
        config["model"]["path"],
    )

    if not model_path.exists():
        pytest.skip(
            "Trained model artifact is not present."
        )

    model = load_model(
        model_path
    )

    assert model is not None
    assert hasattr(
        model,
        "predict",
    )