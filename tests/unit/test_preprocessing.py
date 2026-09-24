import pandas as pd 
from src.preprocessing import (
    build_preprocessor,
    get_feature_columns,
    get_feature_names,
    prepare_xy,
    remove_unused_columns,
    transform_with_saved_preprocessor,
)


def test_prepare_xy():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "is_late": [0, 1, 0],
        }
    )

    X, y = prepare_xy(
        df,
        "is_late",
    )

    assert "is_late" not in X.columns
    assert len(y) == 3


def test_remove_unused_columns():
    X = pd.DataFrame(
        {
            "order_id": ["o1"],
            "customer_id": ["c1"],
            "customer_state": ["SP"],
            "total_price": [100.0],
            "order_purchase_timestamp": [
                "2018-01-01"
            ],
            "order_delivered_customer_date": [
                "2018-01-10"
            ],
        }
    )

    result = remove_unused_columns(
        X,
        id_columns=[
            "order_id",
            "customer_id",
        ],
        future_columns=[
            "order_delivered_customer_date",
        ],
        raw_date_columns=[
            "order_purchase_timestamp",
        ],
    )

    assert list(result.columns) == [
        "customer_state",
        "total_price",
    ]


def test_get_feature_columns():
    X = pd.DataFrame(
        {
            "total_price": [100.0, 200.0],
            "total_items": [1, 2],
            "customer_state": ["SP", "RJ"],
        }
    )

    numerical, categorical = (
        get_feature_columns(
            X,
            ["customer_state"],
        )
    )

    assert "total_price" in numerical
    assert "total_items" in numerical
    assert categorical == ["customer_state"]


def test_preprocessor_fit_and_transform():
    X_train = pd.DataFrame(
        {
            "total_price": [100.0, 200.0, 150.0],
            "customer_state": [
                "SP",
                "RJ",
                "SP",
            ],
        }
    )

    X_val = pd.DataFrame(
        {
            "total_price": [120.0],
            "customer_state": ["MG"],
        }
    )

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

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_val_processed = (
        transform_with_saved_preprocessor(
            preprocessor,
            X_val,
        )
    )

    assert X_train_processed.shape[0] == 3
    assert X_val_processed.shape[0] == 1
    assert (
        X_train_processed.shape[1]
        == X_val_processed.shape[1]
    )


def test_feature_names():
    X = pd.DataFrame(
        {
            "total_price": [100.0, 200.0],
            "customer_state": ["SP", "RJ"],
        }
    )

    numerical, categorical = (
        get_feature_columns(
            X,
            ["customer_state"],
        )
    )

    preprocessor = build_preprocessor(
        numerical,
        categorical,
    )

    preprocessor.fit(X)

    names = get_feature_names(
        preprocessor
    )

    assert len(names) > 0