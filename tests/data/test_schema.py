import pandas as pd


REQUIRED_COLUMNS = {
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "customer_unique_id",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
    "total_items",
    "total_price",
    "total_freight_value",
    "avg_item_price",
    "total_payment_value",
    "max_installments",
    "payment_count",
    "main_payment_type",
    "unique_products",
    "unique_sellers",
}


def test_required_ml_table_columns():
    df = pd.DataFrame(
        {
            column: []
            for column in REQUIRED_COLUMNS
        }
    )

    assert REQUIRED_COLUMNS.issubset(
        set(df.columns)
    )