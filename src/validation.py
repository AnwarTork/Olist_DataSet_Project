import pandas as pd


def get_table_shapes(tables):
    """
    Return number of rows and columns for each table.
    """
    return pd.DataFrame({
        "table": list(tables.keys()),
        "rows": [df.shape[0] for df in tables.values()],
        "columns": [df.shape[1] for df in tables.values()],
    })


def check_unique_key(df, column):
    """
    Check duplicated values in a single-column key.
    """
    return int(df[column].duplicated().sum())


def check_composite_key(df, columns):
    """
    Check duplicated values for a composite key.
    """
    return int(df.duplicated(subset=columns).sum())


def validate_primary_keys(
    orders,
    customers,
    products,
    sellers,
):
    """
    Validate primary keys of the main Olist tables.
    """
    return {
        "orders.order_id": check_unique_key(
            orders, "order_id"
        ),
        "customers.customer_id": check_unique_key(
            customers, "customer_id"
        ),
        "products.product_id": check_unique_key(
            products, "product_id"
        ),
        "sellers.seller_id": check_unique_key(
            sellers, "seller_id"
        ),
    }


def validate_composite_keys(
    order_items,
    order_payments,
):
    """
    Validate composite keys for order items and payments.
    """
    return {
        "order_items.(order_id, order_item_id)": check_composite_key(
            order_items,
            ["order_id", "order_item_id"],
        ),
        "order_payments.(order_id, payment_sequential)": check_composite_key(
            order_payments,
            ["order_id", "payment_sequential"],
        ),
    }


def validate_ml_table(ml_table):
    """
    Validate the final ML table structure.
    """
    results = {
        "rows": len(ml_table),
        "columns": len(ml_table.columns),
        "unique_orders": ml_table["order_id"].nunique(),
        "duplicate_order_ids": ml_table["order_id"].duplicated().sum(),
    }

    return results


def get_missing_values(df):
    """
    Return missing-value counts sorted descending.
    """
    return (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )


def validate_no_duplicate_orders(ml_table):
    """
    Return True if every order_id is unique.
    """
    return ml_table["order_id"].duplicated().sum() == 0