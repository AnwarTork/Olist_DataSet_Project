from src.config import load_config


def test_future_columns_are_defined():
    config = load_config()

    future_columns = (
        config["features"]["future_columns"]
    )

    assert (
        "order_delivered_customer_date"
        in future_columns
    )

    assert (
        "order_delivered_carrier_date"
        in future_columns
    )


def test_identifiers_are_defined():
    config = load_config()

    id_columns = (
        config["features"]["id_columns"]
    )

    assert "order_id" in id_columns
    assert "customer_id" in id_columns
    assert (
        "customer_unique_id"
        in id_columns
    )


def test_raw_purchase_timestamp_is_removed():
    config = load_config()

    raw_date_columns = (
        config["features"]["raw_date_columns"]
    )

    assert (
        "order_purchase_timestamp"
        in raw_date_columns
    )