import sys
from pathlib import Path

import pandas as pd

from src.data_validation import (
    validate_required_columns,
    validate_no_duplicate_orders,
    validate_non_negative_columns,
    validate_target,
    validate_no_leakage_columns,
)


REQUIRED_COLUMNS = [
    "order_id",
    "customer_id",
    "is_late",
]

NON_NEGATIVE_COLUMNS = [
    "total_items",
    "total_price",
    "total_freight_value",
    "total_payment_value",
    "unique_products",
    "unique_sellers",
]

FORBIDDEN_COLUMNS = [
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
]


def validate_dataset(path):
    df = pd.read_csv(path)

    validate_required_columns(
        df,
        REQUIRED_COLUMNS,
    )

    validate_no_duplicate_orders(
        df,
        "order_id",
    )

    validate_non_negative_columns(
        df,
        NON_NEGATIVE_COLUMNS,
    )

    validate_target(
        df,
        "is_late",
    )

    return True


if __name__ == "__main__":

    dataset_path = Path(
        "data/splits/train.csv"
    )

    try:

        validate_dataset(
            dataset_path
        )

        print(
            "Dataset validation passed."
        )

    except Exception as exc:

        print(
            f"Dataset validation failed: {exc}"
        )

        sys.exit(1)