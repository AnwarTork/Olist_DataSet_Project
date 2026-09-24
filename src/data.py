from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import get_database_config


def create_db_engine(config):
    """Create and return a SQLAlchemy PostgreSQL database engine."""

    db_config = get_database_config(config)

    connection_string = (
        "postgresql+psycopg2://"
        f"{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}"
        f"/{db_config['name']}"
    )

    return create_engine(connection_string)


def test_db_connection(engine):
    """Test the database connection."""

    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return result.fetchone()[0] == 1


def get_table_names(engine):
    """Return all public table names from PostgreSQL."""

    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """

    return pd.read_sql(query, engine)


def load_table(engine, table_name):
    """Load one PostgreSQL table into a pandas DataFrame."""

    query = f'SELECT * FROM "{table_name}"'

    return pd.read_sql(query, engine)


def load_all_tables(engine, table_names):
    """Load the required Olist tables."""

    return {
        table_name: load_table(engine, table_name)
        for table_name in table_names
    }


def aggregate_order_items(order_items):
    """Aggregate order-item data to one row per order."""

    items_agg = (
        order_items
        .groupby("order_id")
        .agg(
            total_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            avg_item_price=("price", "mean"),
        )
        .reset_index()
    )

    return items_agg


def aggregate_order_payments(order_payments):
    """Aggregate payment data to one row per order."""

    payments_agg = (
        order_payments
        .groupby("order_id")
        .agg(
            total_payment_value=("payment_value", "sum"),
            max_installments=("payment_installments", "max"),
            payment_count=("payment_sequential", "count"),
        )
        .reset_index()
    )

    payment_type = (
        order_payments
        .groupby("order_id")["payment_type"]
        .agg(
            lambda x: (
                x.mode().iloc[0]
                if not x.mode().empty
                else np.nan
            )
        )
        .reset_index()
        .rename(
            columns={"payment_type": "main_payment_type"}
        )
    )

    payments_agg = payments_agg.merge(
        payment_type,
        on="order_id",
        how="left",
    )

    return payments_agg


def aggregate_order_reviews(order_reviews):
    """Aggregate review data to one row per order."""

    reviews_agg = (
        order_reviews
        .groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "count"),
        )
        .reset_index()
    )

    return reviews_agg


def calculate_unique_products(order_items):
    """Calculate the number of unique products per order."""

    return (
        order_items
        .groupby("order_id")["product_id"]
        .nunique()
        .reset_index()
        .rename(
            columns={"product_id": "unique_products"}
        )
    )


def calculate_unique_sellers(order_items):
    """Calculate the number of unique sellers per order."""

    return (
        order_items
        .groupby("order_id")["seller_id"]
        .nunique()
        .reset_index()
        .rename(
            columns={"seller_id": "unique_sellers"}
        )
    )


def build_ml_table(
    orders,
    customers,
    order_items,
    order_payments,
):
    """
    Build the order-level ML table.

    The resulting table contains one row per order.
    """

    items_agg = aggregate_order_items(order_items)

    payments_agg = aggregate_order_payments(
        order_payments
    )

    unique_products = calculate_unique_products(
        order_items
    )

    unique_sellers = calculate_unique_sellers(
        order_items
    )

    ml_table = orders.copy()

    ml_table = ml_table.merge(
        customers,
        on="customer_id",
        how="left",
    )

    ml_table = ml_table.merge(
        items_agg,
        on="order_id",
        how="left",
    )

    ml_table = ml_table.merge(
        payments_agg,
        on="order_id",
        how="left",
    )

    ml_table = ml_table.merge(
        unique_products,
        on="order_id",
        how="left",
    )

    ml_table = ml_table.merge(
        unique_sellers,
        on="order_id",
        how="left",
    )

    return ml_table


def save_dataframe(df, output_path):
    """Save a DataFrame to CSV."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )