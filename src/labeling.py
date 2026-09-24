import pandas as pd


def load_ml_table(input_path):
    """
    Load the ML table from CSV.
    """
    return pd.read_csv(input_path)


def prepare_label_dates(df):
    """
    Convert delivery-related date columns to datetime.
    """
    df = df.copy()

    date_columns = [
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    return df


def create_labeled_table(df):
    """
    Keep only orders with both actual and estimated
    delivery dates, then create the is_late label.
    """
    df = df.copy()

    labeled_table = df[
        df["order_delivered_customer_date"].notna()
        & df["order_estimated_delivery_date"].notna()
    ].copy()

    labeled_table["is_late"] = (
        labeled_table["order_delivered_customer_date"]
        >
        labeled_table["order_estimated_delivery_date"]
    ).astype(int)

    return labeled_table


def calculate_delivery_delay(df):
    """
    Calculate delivery delay in days.

    Positive value = delivered after estimated date.
    Zero or negative value = delivered on time.
    """
    df = df.copy()

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"]
        -
        df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    return df


def validate_labels(df):
    """
    Validate that is_late matches delivery delay.
    """
    check = (
        (df["delivery_delay_days"] > 0).astype(int)
        == df["is_late"]
    )

    return check.all()


def get_label_distribution(df):
    """
    Return count and percentage for each label.
    """
    class_counts = df["is_late"].value_counts()

    class_percentages = (
        df["is_late"]
        .value_counts(normalize=True)
        * 100
    )

    distribution = pd.DataFrame({
        "count": class_counts,
        "percentage": class_percentages,
    })

    distribution.index = [
        "On Time (0)" if i == 0 else "Late (1)"
        for i in distribution.index
    ]

    return distribution


def calculate_imbalance_ratio(df):
    """
    Calculate majority/minority class ratio.
    """
    counts = df["is_late"].value_counts()

    majority_count = counts.max()
    minority_count = counts.min()

    imbalance_ratio = (
        majority_count / minority_count
    )

    return {
        "majority_count": majority_count,
        "minority_count": minority_count,
        "imbalance_ratio": imbalance_ratio,
    }


def validate_labeled_table(df):
    """
    Validate final labeled dataset.
    """
    return {
        "rows": len(df),
        "unique_orders": df["order_id"].nunique(),
        "duplicate_orders": (
            df["order_id"].duplicated().sum()
        ),
        "missing_labels": df["is_late"].isna().sum(),
    }


def remove_temporary_columns(df):
    """
    Remove columns used only for validation/debugging.
    """
    df = df.copy()

    if "delivery_delay_days" in df.columns:
        df = df.drop(
            columns=["delivery_delay_days"]
        )

    return df


def create_label_pipeline(input_path):
    """
    Complete labeling pipeline.

    This reproduces the logic of Notebook 2:
    1. Load ML table
    2. Convert date columns
    3. Remove rows without required dates
    4. Create is_late
    5. Calculate delivery delay
    6. Validate labels
    7. Remove temporary delay column
    """
    df = load_ml_table(input_path)

    original_rows = len(df)

    df = prepare_label_dates(df)

    labeled_table = create_labeled_table(df)

    labeled_table = calculate_delivery_delay(
        labeled_table
    )

    labels_correct = validate_labels(
        labeled_table
    )

    labeled_table = remove_temporary_columns(
        labeled_table
    )

    return {
        "data": labeled_table,
        "original_rows": original_rows,
        "labeled_rows": len(labeled_table),
        "removed_rows": (
            original_rows - len(labeled_table)
        ),
        "labels_correct": labels_correct,
    }