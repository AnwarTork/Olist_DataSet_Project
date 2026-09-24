import numpy as np
import pandas as pd


def parse_date_columns(df, date_columns):
    df = df.copy()

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def create_date_features(df):
    df = df.copy()

    timestamp = df["order_purchase_timestamp"]

    df["purchase_year"] = timestamp.dt.year
    df["purchase_month"] = timestamp.dt.month
    df["purchase_day"] = timestamp.dt.day
    df["purchase_weekday"] = timestamp.dt.weekday
    df["purchase_hour"] = timestamp.dt.hour

    return df


def create_freight_features(df):
    df = df.copy()

    df["freight_to_price_ratio"] = (
        df["total_freight_value"]
        / df["total_price"].replace(0, np.nan)
    )

    df["freight_per_item"] = (
        df["total_freight_value"]
        / df["total_items"].replace(0, np.nan)
    )

    return df


def create_order_features(df):
    df = df.copy()

    df["items_per_seller"] = (
        df["total_items"]
        / df["unique_sellers"].replace(0, np.nan)
    )

    df["is_multi_seller"] = (
        df["unique_sellers"] > 1
    ).astype(int)

    df["is_weekend"] = (
        df["purchase_weekday"] >= 5
    ).astype(int)

    return df


def create_geographic_features(df):
    df = df.copy()

    if "customer_state" in df.columns and "seller_state" in df.columns:
        df["is_cross_state"] = (
            df["customer_state"] != df["seller_state"]
        ).astype(int)

    return df


def replace_infinite_values(df):
    df = df.copy()

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return df


def create_all_features(df):
    """
    Apply the complete feature-engineering pipeline.
    No fitting is performed here.
    """

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    df = parse_date_columns(df, date_columns)
    df = create_date_features(df)
    df = create_freight_features(df)
    df = create_order_features(df)
    df = create_geographic_features(df)
    df = replace_infinite_values(df)

    return df