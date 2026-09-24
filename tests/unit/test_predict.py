import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression

from src.predict import (
    PredictionError,
    make_prediction,
    prepare_input,
    validate_input,
    validate_required_features,
)

from src.preprocessing import (
    build_preprocessor,
    get_feature_columns,
)


@pytest.fixture
def prediction_objects():
    X_train = pd.DataFrame(
        {
            "total_price": [
                50.0,
                100.0,
                200.0,
                300.0,
            ],
            "total_items": [
                1,
                2,
                3,
                4,
            ],
        }
    )

    y_train = [0, 0, 1, 1]

    numerical, categorical = (
        get_feature_columns(
            X_train,
            [],
        )
    )

    preprocessor = build_preprocessor(
        numerical,
        categorical,
    )

    X_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X_processed,
        y_train,
    )

    return model, preprocessor


def test_validate_input():
    assert validate_input(
        {"total_price": 100}
    )


def test_validate_input_rejects_none():
    with pytest.raises(ValueError):
        validate_input(None)


def test_validate_input_rejects_empty():
    with pytest.raises(ValueError):
        validate_input({})


def test_prepare_input():
    result = prepare_input(
        {
            "total_price": 100.0,
            "total_items": 2,
        }
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert len(result) == 1


def test_required_features():
    data = {
        "total_price": 100.0,
        "total_items": 2,
    }

    assert validate_required_features(
        data,
        [
            "total_price",
            "total_items",
        ],
    )


def test_missing_required_features():
    with pytest.raises(ValueError):
        validate_required_features(
            {"total_price": 100.0},
            [
                "total_price",
                "total_items",
            ],
        )


def test_make_prediction(
    prediction_objects
):
    model, preprocessor = (
        prediction_objects
    )

    result = make_prediction(
        {
            "total_price": 150.0,
            "total_items": 2,
        },
        model,
        preprocessor,
        "1.0.0",
        [
            "total_price",
            "total_items",
        ],
    )

    assert "prediction" in result
    assert "probability" in result

    assert result["prediction"] in [0, 1]

    assert (
        0.0
        <= result["probability"]
        <= 1.0
    )


def test_prediction_does_not_fit_preprocessor(
    prediction_objects
):
    model, preprocessor = (
        prediction_objects
    )

    original_categories = (
        getattr(
            preprocessor,
            "transformers_",
            None,
        )
    )

    make_prediction(
        {
            "total_price": 150.0,
            "total_items": 2,
        },
        model,
        preprocessor,
        "1.0.0",
        [
            "total_price",
            "total_items",
        ],
    )

    assert (
        getattr(
            preprocessor,
            "transformers_",
            None,
        )
        is original_categories
    )