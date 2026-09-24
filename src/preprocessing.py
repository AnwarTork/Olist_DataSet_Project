from pathlib import Path

import joblib
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_xy(
    df,
    target_column
):
    """
    Separate features X and target y.
    """

    y = df[target_column].copy()

    X = df.drop(
        columns=[target_column]
    ).copy()

    return X, y


def remove_unused_columns(
    X,
    id_columns,
    future_columns,
    raw_date_columns
):
    """
    Remove IDs, future information, and raw timestamp columns.
    """

    drop_columns = (
        id_columns
        + future_columns
        + raw_date_columns
    )

    X = X.drop(
        columns=drop_columns,
        errors="ignore"
    )

    return X


def get_feature_columns(
    X,
    categorical_features
):
    """
    Identify numerical and categorical features.
    """

    categorical_features = [
        col
        for col in categorical_features
        if col in X.columns
    ]

    numerical_features = (
        X.select_dtypes(
    include=["number"]
    )
        .columns
        .tolist()
    )

    return (
        numerical_features,
        categorical_features
    )


def build_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Build preprocessing pipeline.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            ),
        ]
    )

    return preprocessor


def fit_and_transform(
    preprocessor,
    X_train,
    X_val,
    X_test
):
    """
    Fit ONLY on training data.

    Validation and test are transformed
    using the fitted training preprocessor.
    """

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed
    )


def get_feature_names(preprocessor):
    """
    Get final transformed feature names.
    """

    return preprocessor.get_feature_names_out()


def save_artifacts(
    preprocessor,
    feature_names,
    numerical_features,
    categorical_features,
    output_dir
):
    """
    Save all preprocessing artifacts.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        output_dir / "preprocessor.joblib"
    )

    joblib.dump(
        feature_names,
        output_dir / "feature_names.joblib"
    )

    feature_metadata = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "final_feature_names": feature_names.tolist(),
    }

    joblib.dump(
        feature_metadata,
        output_dir / "feature_metadata.joblib"
    )


def save_processed_data(
    X_train_processed,
    X_val_processed,
    X_test_processed,
    y_train,
    y_val,
    y_test,
    output_dir
):
    """
    Save processed matrices and target variables.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        X_train_processed,
        output_dir / "X_train_processed.joblib"
    )

    joblib.dump(
        X_val_processed,
        output_dir / "X_val_processed.joblib"
    )

    joblib.dump(
        X_test_processed,
        output_dir / "X_test_processed.joblib"
    )

    joblib.dump(
        y_train,
        output_dir / "y_train.joblib"
    )

    joblib.dump(
        y_val,
        output_dir / "y_val.joblib"
    )

    joblib.dump(
        y_test,
        output_dir / "y_test.joblib"
    )


def load_preprocessor(
    preprocessor_path
):
    """
    Load an already fitted preprocessor.
    """

    return joblib.load(
        preprocessor_path
    )


def transform_with_saved_preprocessor(
    preprocessor,
    X
):
    """
    Transform new data without fitting.
    """

    return preprocessor.transform(X)