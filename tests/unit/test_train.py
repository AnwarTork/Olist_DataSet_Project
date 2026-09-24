import numpy as np

from src.train import (
    calculate_scale_pos_weight,
    load_model,
    predict_model,
    save_model,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)


def create_training_data():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ]
    )

    y = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1]
    )

    return X, y


def test_scale_pos_weight():
    _, y = create_training_data()

    weight = calculate_scale_pos_weight(
        y
    )

    assert weight == 5 / 3


def test_logistic_regression():
    X, y = create_training_data()

    model = train_logistic_regression(
        X,
        y,
    )

    predictions, probabilities = (
        predict_model(
            model,
            X,
        )
    )

    assert len(predictions) == len(y)
    assert len(probabilities) == len(y)

    assert set(
        predictions
    ).issubset({0, 1})


def test_random_forest():
    X, y = create_training_data()

    model = train_random_forest(
        X,
        y,
        n_estimators=10,
    )

    predictions, probabilities = (
        predict_model(
            model,
            X,
        )
    )

    assert len(predictions) == len(y)
    assert len(probabilities) == len(y)


def test_xgboost():
    X, y = create_training_data()

    model = train_xgboost(
        X,
        y,
        n_estimators=10,
        max_depth=2,
    )

    predictions, probabilities = (
        predict_model(
            model,
            X,
        )
    )

    assert len(predictions) == len(y)
    assert len(probabilities) == len(y)


def test_save_and_load_model(
    tmp_path
):
    X, y = create_training_data()

    model = train_logistic_regression(
        X,
        y,
    )

    path = (
        tmp_path
        / "model.joblib"
    )

    save_model(
        model,
        path,
    )

    loaded_model = load_model(
        path,
    )

    predictions, _ = (
        predict_model(
            loaded_model,
            X,
        )
    )

    assert len(predictions) == len(y)