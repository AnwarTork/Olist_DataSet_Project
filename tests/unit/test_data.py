import pandas as pd

from src.data import (
    aggregate_order_items,
    aggregate_order_payments,
    build_ml_table,
    calculate_unique_products,
    calculate_unique_sellers,
)


def test_aggregate_order_items():
    data = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "order_item_id": [1, 2, 1],
            "price": [100.0, 50.0, 200.0],
            "freight_value": [10.0, 5.0, 20.0],
        }
    )

    result = aggregate_order_items(data)

    assert len(result) == 2

    row = result[
        result["order_id"] == "o1"
    ].iloc[0]

    assert row["total_items"] == 2
    assert row["total_price"] == 150.0
    assert row["total_freight_value"] == 15.0
    assert row["avg_item_price"] == 75.0


def test_aggregate_order_payments():
    data = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "payment_value": [
                100.0,
                50.0,
                200.0,
            ],
            "payment_installments": [
                2,
                5,
                1,
            ],
            "payment_sequential": [
                1,
                2,
                1,
            ],
            "payment_type": [
                "credit_card",
                "credit_card",
                "boleto",
            ],
        }
    )

    result = aggregate_order_payments(data)

    row = result[
        result["order_id"] == "o1"
    ].iloc[0]

    assert row["total_payment_value"] == 150.0
    assert row["max_installments"] == 5
    assert row["payment_count"] == 2
    assert row["main_payment_type"] == "credit_card"


def test_unique_products():
    data = pd.DataFrame(
        {
            "order_id": [
                "o1",
                "o1",
                "o1",
                "o2",
            ],
            "product_id": [
                "p1",
                "p1",
                "p2",
                "p3",
            ],
        }
    )

    result = calculate_unique_products(data)

    row = result[
        result["order_id"] == "o1"
    ].iloc[0]

    assert row["unique_products"] == 2


def test_unique_sellers():
    data = pd.DataFrame(
        {
            "order_id": [
                "o1",
                "o1",
                "o1",
                "o2",
            ],
            "seller_id": [
                "s1",
                "s1",
                "s2",
                "s3",
            ],
        }
    )

    result = calculate_unique_sellers(data)

    row = result[
        result["order_id"] == "o1"
    ].iloc[0]

    assert row["unique_sellers"] == 2


def test_build_ml_table_has_one_row_per_order():
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
        }
    )

    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "customer_city": ["city1", "city2"],
            "customer_state": ["SP", "RJ"],
        }
    )

    order_items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "order_item_id": [1, 2, 1],
            "price": [100.0, 50.0, 200.0],
            "freight_value": [10.0, 5.0, 20.0],
            "product_id": ["p1", "p2", "p3"],
            "seller_id": ["s1", "s1", "s2"],
        }
    )

    order_payments = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "payment_value": [
                100.0,
                50.0,
                200.0,
            ],
            "payment_installments": [
                2,
                5,
                1,
            ],
            "payment_sequential": [
                1,
                2,
                1,
            ],
            "payment_type": [
                "credit_card",
                "credit_card",
                "boleto",
            ],
        }
    )

    result = build_ml_table(
        orders,
        customers,
        order_items,
        order_payments,
    )

    assert len(result) == len(orders)
    assert result["order_id"].is_unique