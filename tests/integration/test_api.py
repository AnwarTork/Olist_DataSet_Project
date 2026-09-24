import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.predict import make_prediction
from src.preprocessing import (
    build_preprocessor,
    get_feature_columns,
)


def test_prediction_pipeline_end_to_end():
    X_train = pd.DataFrame(
        {
            "total_price": [
                50.0,
                80.0,
                120.0,
                200.0,
                250.0,
                300.0,
            ],
            "total_items": [
                1,
                1,
                2,
                2,
                3,
                4,
            ],
            "customer_state": [
                "SP",
                "SP",
                "RJ",
                "RJ",
                "MG",
                "MG",
            ],
        }
    )

    y_train = [
        0,
        0,
        0,
        1,
        1,
        1,
    ]

    numerical, categorical = (
        get_feature_columns(
            X_train,
            ["customer_state"],
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

    result = make_prediction(
        {
            "total_price": 150.0,
            "total_items": 2,
            "customer_state": "SP",
        },
        model,
        preprocessor,
        "1.0.0",
        [
            "total_price",
            "total_items",
            "customer_state",
        ],
    )

    assert result["prediction"] in [
        0,
        1,
    ]

    assert (
        0.0
        <= result["probability"]
        <= 1.0
    )