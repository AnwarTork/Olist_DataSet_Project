import pandas as pd
from sklearn.model_selection import train_test_split


def load_labeled_data(input_path):
    """
    Load the labeled ML dataset.
    """
    return pd.read_csv(input_path)


def validate_dataset(df):
    """
    Validate labels and order uniqueness.
    """
    return {
        "missing_labels": int(df["is_late"].isna().sum()),
        "duplicate_orders": int(
            df["order_id"].duplicated().sum()
        ),
        "rows": len(df),
    }


def prepare_purchase_date(df):
    """
    Convert order purchase timestamp to datetime
    and temporarily create purchase_year for inspection.
    """
    df = df.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df["purchase_year"] = (
        df["order_purchase_timestamp"].dt.year
    )

    return df


def split_data(
    df,
    test_size=0.30,
    validation_size=0.50,
    random_state=42,
):
    """
    Split the dataset into train, validation and test.

    First:
        70% train
        30% temporary

    Then:
        50% validation
        50% test

    Final:
        70% train
        15% validation
        15% test

    Stratification is performed using is_late.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["is_late"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=validation_size,
        random_state=random_state,
        stratify=temp_df["is_late"],
    )

    return train_df, val_df, test_df


def get_label_distribution(data):
    """
    Return label counts and percentages.
    """
    counts = data["is_late"].value_counts()

    percentages = (
        data["is_late"]
        .value_counts(normalize=True)
        * 100
    )

    return {
        "rows": len(data),
        "on_time_count": int(counts.get(0, 0)),
        "on_time_percentage": float(
            percentages.get(0, 0)
        ),
        "late_count": int(counts.get(1, 0)),
        "late_percentage": float(
            percentages.get(1, 0)
        ),
    }


def get_late_ratio(data):
    """
    Return the proportion of late orders.
    """
    return float(data["is_late"].mean())


def check_order_overlap(
    train_df,
    val_df,
    test_df,
):
    """
    Check whether any order_id exists in more than one split.
    """
    train_orders = set(train_df["order_id"])
    val_orders = set(val_df["order_id"])
    test_orders = set(test_df["order_id"])

    return {
        "train_validation": len(
            train_orders & val_orders
        ),
        "train_test": len(
            train_orders & test_orders
        ),
        "validation_test": len(
            val_orders & test_orders
        ),
    }


def get_date_range(data):
    """
    Return minimum and maximum purchase timestamp.
    """
    return {
        "min": data["order_purchase_timestamp"].min(),
        "max": data["order_purchase_timestamp"].max(),
    }


def remove_temporary_columns(
    train_df,
    val_df,
    test_df,
):
    """
    Remove temporary inspection columns.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for data in [train_df, val_df, test_df]:
        if "purchase_year" in data.columns:
            data.drop(
                columns=["purchase_year"],
                inplace=True,
            )

    return train_df, val_df, test_df


def validate_splits(
    train_df,
    val_df,
    test_df,
    original_df,
):
    """
    Validate final train/validation/test splits.
    """
    total_rows = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    overlaps = check_order_overlap(
        train_df,
        val_df,
        test_df,
    )

    return {
        "train_shape": train_df.shape,
        "validation_shape": val_df.shape,
        "test_shape": test_df.shape,
        "total_rows": total_rows,
        "original_rows": len(original_df),
        "row_count_matches": (
            total_rows == len(original_df)
        ),
        "train_validation_overlap": overlaps[
            "train_validation"
        ],
        "train_test_overlap": overlaps[
            "train_test"
        ],
        "validation_test_overlap": overlaps[
            "validation_test"
        ],
    }


def create_splits(
    input_path,
    test_size=0.30,
    validation_size=0.50,
    random_state=42,
):
    """
    Complete train/validation/test splitting pipeline.
    """
    df = load_labeled_data(input_path)

    original_df = df.copy()

    validation = validate_dataset(df)

    df = prepare_purchase_date(df)

    train_df, val_df, test_df = split_data(
        df=df,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    train_df, val_df, test_df = remove_temporary_columns(
        train_df,
        val_df,
        test_df,
    )

    split_validation = validate_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        original_df=original_df,
    )

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "dataset_validation": validation,
        "split_validation": split_validation,
    }