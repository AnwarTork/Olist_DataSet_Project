import numpy as np
import pandas as pd


def get_dataset_info(df):
    """
    Return basic information about the dataset.
    """
    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_mb": memory_mb,
    }

# def summarize_dataframe(df):
#     """Return basic dataframe summary."""

#     return {
#         "rows": len(df),
#         "columns": len(df.columns),
#         "duplicates": int(df.duplicated().sum()),
#         "missing_values": (
#             int(df.isna().sum().sum())
#         ),
#     }

def label_distribution(
    df,
    target_column="is_late",
):
    """Return class counts and proportions."""

    counts = df[target_column].value_counts(
        dropna=False
    )

    proportions = df[target_column].value_counts(
        normalize=True,
        dropna=False,
    )

    return pd.DataFrame(
        {
            "count": counts,
            "proportion": proportions,
        }
    )
def group_by_label(
    df,
    column,
    target_column="is_late",
):
    """Calculate label distribution by a categorical column."""

    return pd.crosstab(
        df[column],
        df[target_column],
        normalize="index",
    )

def date_range(
    df,
    date_column="order_purchase_timestamp",
):
    """Return minimum and maximum dates."""

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    return {
        "min": dates.min(),
        "max": dates.max(),
    }

def get_missing_summary(df):
    """
    Return missing-value counts and percentages.
    """
    missing = df.isna().sum()

    missing = (
        missing[missing > 0]
        .sort_values(ascending=False)
    )

    return pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": (
            missing / len(df) * 100
        ),
    })


def get_column_types(
    df,
    id_columns,
    categorical_columns,
    date_columns,
    target_column,
):
    """
    Classify dataset columns for EDA.
    """
    numerical_columns = (
        df.select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    numerical_columns = [
        col
        for col in numerical_columns
        if col != target_column
    ]

    text_columns = []

    return {
        "id": id_columns,
        "categorical": categorical_columns,
        "date": date_columns,
        "numerical": numerical_columns,
        "text": text_columns,
        "target": target_column,
    }


def get_numeric_statistics(
    df,
    numerical_columns,
):
    """
    Return descriptive statistics for numerical features.
    """
    return df[numerical_columns].describe().T


def get_skewness(
    df,
    numerical_columns,
):
    """
    Return numerical feature skewness.
    """
    return (
        df[numerical_columns]
        .skew()
        .sort_values(ascending=False)
    )


def get_outlier_summary(
    df,
    numerical_columns,
):
    """
    Calculate IQR-based outlier statistics.
    """
    outlier_summary = []

    for col in numerical_columns:
        series = df[col].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = (
            (series < lower)
            | (series > upper)
        ).sum()

        outlier_percentage = (
            outliers / len(series) * 100
        )

        outlier_summary.append({
            "feature": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outliers,
            "outlier_percentage": (
                outlier_percentage
            ),
        })

    return (
        pd.DataFrame(outlier_summary)
        .sort_values(
            "outlier_percentage",
            ascending=False,
        )
    )


def find_negative_values(
    df,
    numerical_columns,
):
    """
    Find negative values in numerical features.
    """
    negative_values = {}

    for col in numerical_columns:
        count = int((df[col] < 0).sum())

        if count > 0:
            negative_values[col] = count

    return negative_values


def get_categorical_summary(
    df,
    categorical_columns,
):
    """
    Return cardinality and missing-value summary.
    """
    summary = []

    for col in categorical_columns:
        summary.append({
            "feature": col,
            "unique_values": df[col].nunique(),
            "missing": df[col].isna().sum(),
        })

    return pd.DataFrame(summary)


def get_category_counts(
    df,
    categorical_columns,
    top_n=20,
):
    """
    Return top category counts for categorical columns.
    """
    results = {}

    for col in categorical_columns:
        results[col] = (
            df[col]
            .value_counts(
                dropna=False
            )
            .head(top_n)
        )

    return results


def get_rare_categories(
    df,
    categorical_columns,
    threshold=1.0,
):
    """
    Find categories representing less than
    the given percentage of observations.
    """
    results = {}

    for col in categorical_columns:
        counts = (
            df[col]
            .value_counts(
                normalize=True,
                dropna=False,
            )
            * 100
        )

        results[col] = counts[
            counts < threshold
        ]

    return results


def compare_normalized_categories(
    df,
    categorical_columns,
):
    """
    Compare original and normalized category cardinality.
    """
    results = []

    for col in categorical_columns:
        if df[col].dtype == "object":
            cleaned = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
            )

            results.append({
                "feature": col,
                "original_unique": (
                    df[col].nunique()
                ),
                "normalized_unique": (
                    cleaned.nunique()
                ),
            })

    return pd.DataFrame(results)


def get_numeric_label_summary(
    df,
    numerical_columns,
    target_column="is_late",
):
    """
    Compare numerical feature means by target label.
    """
    summary = (
        df
        .groupby(target_column)[numerical_columns]
        .mean()
        .T
    )

    summary.columns = [
        "On_Time" if c == 0 else "Late"
        for c in summary.columns
    ]

    return summary


def get_state_late_rate(
    df,
    state_column="customer_state",
    target_column="is_late",
):
    """
    Calculate late-delivery rate by customer state.
    """
    return (
        df
        .groupby(state_column)[target_column]
        .mean()
        .sort_values(ascending=False)
        * 100
    )


def get_payment_late_rate(
    df,
    payment_column="main_payment_type",
    target_column="is_late",
):
    """
    Calculate late-delivery rate by payment type.
    """
    result = (
        df
        .groupby(payment_column)[target_column]
        .agg(["mean", "count"])
    )

    result["late_percentage"] = (
        result["mean"] * 100
    )

    return result.sort_values(
        "late_percentage",
        ascending=False,
    )


def get_payment_crosstab(
    df,
    payment_column="main_payment_type",
    target_column="is_late",
):
    """
    Create payment type vs label contingency table.
    """
    return pd.crosstab(
        df[payment_column],
        df[target_column],
    )


def get_correlation_with_target(
    df,
    numerical_columns,
    target_column="is_late",
):
    """
    Calculate Pearson correlation between
    numerical features and target.
    """
    correlation_matrix = df[
        numerical_columns + [target_column]
    ].corr()

    return (
        correlation_matrix[target_column]
        .drop(target_column)
        .sort_values()
    )


def get_orders_over_time(
    df,
    timestamp_column="order_purchase_timestamp",
    frequency="ME",
):
    """
    Calculate number of orders over time.
    """
    temp = df.set_index(timestamp_column)

    return temp.resample(frequency).size()


def get_late_rate_over_time(
    df,
    timestamp_column="order_purchase_timestamp",
    target_column="is_late",
    frequency="ME",
):
    """
    Calculate late-delivery rate over time.
    """
    temp = df.set_index(timestamp_column)

    return (
        temp[target_column]
        .resample(frequency)
        .mean()
        * 100
    )


def get_weekday_late_rate(
    df,
    timestamp_column="order_purchase_timestamp",
    target_column="is_late",
):
    """
    Calculate late rate by purchase weekday.
    """
    temp = df.copy()

    temp["purchase_weekday"] = (
        temp[timestamp_column]
        .dt.day_name()
    )

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    return (
        temp
        .groupby("purchase_weekday")[target_column]
        .mean()
        .reindex(weekday_order)
        * 100
    )


def get_month_late_rate(
    df,
    timestamp_column="order_purchase_timestamp",
    target_column="is_late",
):
    """
    Calculate late rate by purchase month.
    """
    temp = df.copy()

    temp["purchase_month"] = (
        temp[timestamp_column].dt.month
    )

    return (
        temp
        .groupby("purchase_month")[target_column]
        .mean()
        * 100
    )


def add_delivery_days(
    df,
    purchase_column="order_purchase_timestamp",
    delivery_column="order_delivered_customer_date",
):
    """
    Calculate delivery duration in days.
    """
    temp = df.copy()

    temp["delivery_days"] = (
        temp[delivery_column]
        - temp[purchase_column]
    ).dt.total_seconds() / 86400

    return temp


def get_delivery_days_by_label(
    df,
    delivery_days_column="delivery_days",
    target_column="is_late",
):
    """
    Return delivery-time statistics by label.
    """
    return (
        df
        .groupby(target_column)[delivery_days_column]
        .describe()
    )


def get_customer_state_summary(
    df,
    state_column="customer_state",
    target_column="is_late",
):
    """
    Return order count and late rate by customer state.
    """
    summary = (
        df
        .groupby(state_column)
        .agg(
            orders=("order_id", "count"),
            late_rate=(target_column, "mean"),
        )
    )

    summary["late_rate"] *= 100

    return summary.sort_values(
        "late_rate",
        ascending=False,
    )


def create_geo_by_zip(geolocation):
    """
    Aggregate geographic coordinates by ZIP prefix.
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


def create_seller_order_data(
    order_items,
    sellers,
):
    """
    Join order items with seller information.
    """
    return (
        order_items[
            ["order_id", "seller_id"]
        ]
        .merge(
            sellers[
                [
                    "seller_id",
                    "seller_zip_code_prefix",
                    "seller_city",
                    "seller_state",
                ]
            ],
            on="seller_id",
            how="left",
        )
    )


def get_seller_statistics(
    seller_order_data,
    sellers,
):
    """
    Calculate seller-level geographic summaries.
    """
    seller_count_per_order = (
        seller_order_data
        .groupby("order_id")["seller_id"]
        .nunique()
    )

    seller_state_summary = (
        seller_order_data
        .groupby("seller_state")["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    seller_state_count = (
        sellers
        .groupby("seller_state")["seller_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    return {
        "orders_with_multiple_sellers": int(
            (seller_count_per_order > 1).sum()
        ),
        "maximum_sellers_per_order": int(
            seller_count_per_order.max()
        ),
        "orders_by_seller_state": (
            seller_state_summary
        ),
        "sellers_by_state": seller_state_count,
    }


def get_seller_state_late_rate(
    seller_order_data,
    train_df,
    target_column="is_late",
):
    """
    Calculate late-delivery rate by seller state.
    """
    train_seller_data = (
        seller_order_data
        .merge(
            train_df[
                ["order_id", target_column]
            ],
            on="order_id",
            how="inner",
        )
    )

    return (
        train_seller_data
        .groupby("seller_state")[target_column]
        .mean()
        .sort_values(ascending=False)
        * 100
    )


def create_customer_geo(
    train_df,
    geo_by_zip,
):
    """
    Attach coordinates to customers.
    """
    customer_geo = (
        train_df[
            [
                "order_id",
                "customer_zip_code_prefix",
            ]
        ]
        .merge(
            geo_by_zip,
            left_on="customer_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left",
        )
    )

    return customer_geo.rename(
        columns={
            "latitude": "customer_lat",
            "longitude": "customer_lng",
        }
    )


def create_seller_geo(
    seller_order_data,
    geo_by_zip,
):
    """
    Attach coordinates to sellers.
    """
    seller_geo = (
        seller_order_data[
            [
                "order_id",
                "seller_zip_code_prefix",
            ]
        ]
        .drop_duplicates()
        .merge(
            geo_by_zip,
            left_on="seller_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left",
        )
    )

    return seller_geo.rename(
        columns={
            "latitude": "seller_lat",
            "longitude": "seller_lng",
        }
    )


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate great-circle distance in kilometers.
    """
    earth_radius_km = 6371

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

    return earth_radius_km * c


def calculate_order_distances(
    train_df,
    seller_order_data,
    geo_by_zip,
):
    """
    Calculate customer-seller distance per order.
    """
    customer_geo = create_customer_geo(
        train_df,
        geo_by_zip,
    )

    seller_geo = create_seller_geo(
        seller_order_data,
        geo_by_zip,
    )

    geo_orders = customer_geo.merge(
        seller_geo,
        on="order_id",
        how="inner",
    )

    geo_orders["distance_km"] = (
        haversine_distance(
            geo_orders["customer_lat"],
            geo_orders["customer_lng"],
            geo_orders["seller_lat"],
            geo_orders["seller_lng"],
        )
    )

    distance_by_order = (
        geo_orders
        .groupby("order_id")["distance_km"]
        .mean()
        .reset_index()
    )

    distance_by_order = (
        distance_by_order
        .merge(
            train_df[
                ["order_id", "is_late"]
            ],
            on="order_id",
            how="inner",
        )
    )

    return geo_orders, distance_by_order