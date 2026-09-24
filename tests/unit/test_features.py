import numpy as np
import pandas as pd

from src.features import (
    add_date_features,
    add_order_features,
    add_product_features,
    create_features,
)


def test_add_date_features():
    df = pd.DataFrame(
        {
            "order_id": ["o1"],
            "order_purchase_timestamp": [
                "2018-01-06 15:30:00"
            ],
        }
    )

    result = add_date_features(df)

    assert result.loc[0, "purchase_year"] == 2018
    assert result.loc[0, "purchase_month"] == 1
    assert result.loc[0, "purchase_day"] == 6
    assert result.loc[0, "purchase_weekday"] == 5
    assert result.loc[0, "purchase_hour"] == 15
    assert result.loc[0, "is_weekend"] == 1


def test_add_order_features():
    df = pd.DataFrame(
        {
            "unique_sellers": [2],
            "total_freight_value": [20.0],
            "total_price": [100.0],
            "total_items": [4],
        }
    )

    result = add_order_features(df)

    assert result.loc[0, "multi_seller"] == 1
    assert result.loc[0, "freight_ratio"] == 0.2
    assert result.loc[0, "freight_per_item"] == 5.0


def test_add_product_features():
    ml_table = pd.DataFrame(
        {
            "order_id": ["o1"],
        }
    )

    order_items = pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "product_id": ["p1", "p2"],
        }
    )

    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_weight_g": [1000.0, 2000.0],
            "product_length_cm": [10.0, 20.0],
            "product_height_cm": [10.0, 20.0],
            "product_width_cm": [10.0, 20.0],
        }
    )

    result = add_product_features(
        ml_table,
        order_items,
        products,
    )

    assert result.loc[0, "total_weight"] == 3000.0
    assert result.loc[0, "total_volume"] == 9000.0
    assert result.loc[0, "avg_volume"] == 4500.0
    assert result.loc[0, "max_volume"] == 8000.0


def test_create_features_preserves_one_row_per_order():
    ml_table = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "order_purchase_timestamp": [
                "2018-01-06 15:30:00",
                "2018-02-10 10:00:00",
            ],
            "unique_sellers": [1, 2],
            "total_freight_value": [
                10.0,
                20.0,
            ],
            "total_price": [
                100.0,
                200.0,
            ],
            "total_items": [1, 2],
            "customer_zip_code_prefix": [
                1000,
                2000,
            ],
            "customer_state": [
                "SP",
                "RJ",
            ],
        }
    )

    order_items = pd.DataFrame(
        {
            "order_id": [
                "o1",
                "o2",
            ],
            "product_id": [
                "p1",
                "p2",
            ],
            "seller_id": [
                "s1",
                "s2",
            ],
        }
    )

    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_weight_g": [
                1000.0,
                2000.0,
            ],
            "product_length_cm": [
                10.0,
                20.0,
            ],
            "product_height_cm": [
                10.0,
                20.0,
            ],
            "product_width_cm": [
                10.0,
                20.0,
            ],
        }
    )

    sellers = pd.DataFrame(
        {
            "seller_id": ["s1", "s2"],
            "seller_zip_code_prefix": [
                1000,
                2000,
            ],
            "seller_state": [
                "SP",
                "SP",
            ],
        }
    )

    geolocation = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [
                1000,
                2000,
            ],
            "geolocation_lat": [
                -23.5,
                -22.9,
            ],
            "geolocation_lng": [
                -46.6,
                -43.2,
            ],
        }
    )

    result = create_features(
        ml_table,
        order_items,
        products,
        sellers,
        geolocation,
    )

    assert len(result) == len(ml_table)
    assert result["order_id"].is_unique
    assert "purchase_month" in result.columns
    assert "total_weight" in result.columns
    assert "distance_km" in result.columns
    assert "is_cross_state" in result.columns