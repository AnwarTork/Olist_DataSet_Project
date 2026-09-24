from pathlib import Path

import pandas as pd


def load_data(path):
    """Load a CSV dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def validate_required_columns(
    df,
    required_columns,
):
    """Validate required columns."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return True


def validate_no_duplicate_orders(
    df,
    order_column="order_id",
):
    """Ensure one row per order."""

    if order_column not in df.columns:
        raise ValueError(
            f"Missing column: {order_column}"
        )

    duplicates = int(
        df[order_column].duplicated().sum()
    )

    if duplicates > 0:
        raise ValueError(
            f"Found {duplicates} duplicate orders."
        )

    return True


def validate_non_negative_columns(
    df,
    columns,
):
    """Ensure selected numeric columns are non-negative."""

    for column in columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

        invalid = (
            df[column]
            .dropna()
            < 0
        ).sum()

        if invalid > 0:
            raise ValueError(
                f"{column} contains "
                f"{invalid} negative values."
            )

    return True


def validate_missing_rates(
    df,
    max_missing_rates,
):
    """Validate maximum allowed missing rates."""

    for column, maximum in (
        max_missing_rates.items()
    ):

        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

        missing_rate = (
            df[column].isna().mean()
        )

        if missing_rate > maximum:
            raise ValueError(
                f"{column} missing rate "
                f"{missing_rate:.4f} exceeds "
                f"allowed {maximum:.4f}."
            )

    return True


def validate_no_leakage_columns(
    feature_columns,
    forbidden_columns,
):
    """Ensure future/leakage columns are absent."""

    leakage_columns = [
        column
        for column in forbidden_columns
        if column in feature_columns
    ]

    if leakage_columns:
        raise ValueError(
            "Leakage columns found in features: "
            + ", ".join(leakage_columns)
        )

    return True


def validate_target(
    df,
    target_column="is_late",
):
    """Validate binary target."""

    if target_column not in df.columns:
        raise ValueError(
            f"Missing target: {target_column}"
        )

    if df[target_column].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    values = set(
        df[target_column].unique()
    )

    if not values.issubset({0, 1}):
        raise ValueError(
            f"Target must contain only "
            f"0 and 1. Found: {values}"
        )

    return True
