import numpy as np
import pandas as pd


def add_date_features(ml_table):
    """
    Add date-based features from order_purchase_timestamp.
    """
    df = ml_table.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df["purchase_year"] = (
        df["order_purchase_timestamp"].dt.year
    )

    df["purchase_month"] = (
        df["order_purchase_timestamp"].dt.month
    )

    df["purchase_day"] = (
        df["order_purchase_timestamp"].dt.day
    )

    df["purchase_weekday"] = (
        df["order_purchase_timestamp"].dt.weekday
    )

    df["purchase_hour"] = (
        df["order_purchase_timestamp"].dt.hour
    )

    df["is_weekend"] = (
        df["purchase_weekday"] >= 5
    ).astype(int)

    return df


def add_order_features(ml_table):
    """
    Add order-level engineered features.
    """
    df = ml_table.copy()

    df["multi_seller"] = (
        df["unique_sellers"] > 1
    ).astype(int)

    df["freight_ratio"] = (
        df["total_freight_value"]
        / df["total_price"].replace(0, np.nan)
    )

    df["freight_per_item"] = (
        df["total_freight_value"]
        / df["total_items"].replace(0, np.nan)
    )

    return df


def add_product_features(ml_table, order_items, products):
    """
    Add product weight and volume features.
    """
    df = ml_table.copy()
    products_df = products.copy()

    products_df["product_volume_cm3"] = (
        products_df["product_length_cm"]
        * products_df["product_height_cm"]
        * products_df["product_width_cm"]
    )

    order_product_data = order_items[
        ["order_id", "product_id"]
    ].merge(
        products_df[
            [
                "product_id",
                "product_weight_g",
                "product_volume_cm3",
            ]
        ],
        on="product_id",
        how="left",
    )

    product_features = (
        order_product_data
        .groupby("order_id")
        .agg(
            total_weight=("product_weight_g", "sum"),
            total_volume=("product_volume_cm3", "sum"),
            avg_volume=("product_volume_cm3", "mean"),
            max_volume=("product_volume_cm3", "max"),
        )
        .reset_index()
    )

    df = df.merge(
        product_features,
        on="order_id",
        how="left",
    )

    return df


def create_geo_by_zip(geolocation):
    """
    Aggregate geolocation coordinates by ZIP prefix.
    """
    return (
        geolocation
        .groupby("geolocation_zip_code_prefix")
        .agg(
            latitude=("geolocation_lat", "mean"),
            longitude=("geolocation_lng", "mean"),
        )
        .reset_index()
    )


def create_customer_geo(ml_table, geo_by_zip):
    """
    Attach geographic coordinates to customers.
    """
    customer_geo = ml_table[
        [
            "order_id",
            "customer_zip_code_prefix",
        ]
    ].merge(
        geo_by_zip,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )

    customer_geo = customer_geo.rename(
        columns={
            "latitude": "customer_lat",
            "longitude": "customer_lng",
        }
    )

    return customer_geo


def create_seller_geo(
    order_items,
    sellers,
    geo_by_zip,
):
    """
    Attach geographic coordinates to sellers.
    """
    seller_order_data = (
        order_items[
            ["order_id", "seller_id"]
        ].merge(
            sellers[
                [
                    "seller_id",
                    "seller_zip_code_prefix",
                    "seller_state",
                ]
            ],
            on="seller_id",
            how="left",
        )
    )

    seller_geo = seller_order_data.merge(
        geo_by_zip,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )

    seller_geo = seller_geo.rename(
        columns={
            "latitude": "seller_lat",
            "longitude": "seller_lng",
        }
    )

    return seller_geo


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate great-circle distance in kilometers.
    """
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return R * c


def add_geographic_features(
    ml_table,
    order_items,
    sellers,
    geolocation,
):
    """
    Add distance and cross-state features.
    """
    df = ml_table.copy()

    geo_by_zip = create_geo_by_zip(
        geolocation
    )

    customer_geo = create_customer_geo(
        df,
        geo_by_zip,
    )

    seller_geo = create_seller_geo(
        order_items,
        sellers,
        geo_by_zip,
    )

    geo_orders = customer_geo.merge(
        seller_geo[
            [
                "order_id",
                "seller_id",
                "seller_state",
                "seller_lat",
                "seller_lng",
            ]
        ],
        on="order_id",
        how="left",
    )

    geo_orders["distance_km"] = haversine_distance(
        geo_orders["customer_lat"],
        geo_orders["customer_lng"],
        geo_orders["seller_lat"],
        geo_orders["seller_lng"],
    )

    geo_orders = geo_orders.merge(
        df[
            [
                "order_id",
                "customer_state",
            ]
        ],
        on="order_id",
        how="left",
    )

    geo_orders["is_cross_state"] = (
        geo_orders["customer_state"]
        != geo_orders["seller_state"]
    ).astype(int)

    geo_features = (
        geo_orders
        .groupby("order_id")
        .agg(
            distance_km=("distance_km", "mean"),
            is_cross_state=("is_cross_state", "max"),
        )
        .reset_index()
    )

    df = df.merge(
        geo_features,
        on="order_id",
        how="left",
    )

    return df


def create_features(
    ml_table,
    order_items,
    products,
    sellers,
    geolocation,
):
    """
    Apply all feature engineering steps
    used in the original Notebook 1.
    """
    df = ml_table.copy()

    df = add_date_features(df)

    df = add_order_features(df)

    df = add_product_features(
        df,
        order_items,
        products,
    )

    df = add_geographic_features(
        df,
        order_items,
        sellers,
        geolocation,
    )

    return df