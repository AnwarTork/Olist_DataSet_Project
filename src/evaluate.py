import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


def calculate_metrics(
    y_true,
    y_pred,
    y_proba
):
    """
    Calculate classification metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_proba
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_proba
        )
    }


def evaluate_model(
    model,
    X,
    y
):
    """
    Generate predictions and calculate metrics.
    """

    y_pred = model.predict(X)

    y_proba = model.predict_proba(
        X
    )[:, 1]

    metrics = calculate_metrics(
        y,
        y_pred,
        y_proba
    )

    return (
        y_pred,
        y_proba,
        metrics
    )


def get_classification_report(
    y_true,
    y_pred
):
    return classification_report(
        y_true,
        y_pred,
        target_names=[
            "On-time",
            "Late"
        ],
        zero_division=0
    )


def get_confusion_matrix(
    y_true,
    y_pred
):
    return confusion_matrix(
        y_true,
        y_pred
    )


def metrics_to_dataframe(
    metrics,
    model_name,
    **model_parameters
):
    row = {
        "model": model_name,
        **model_parameters,
        **metrics
    }

    return pd.DataFrame(
        [row]
    )


def confusion_matrix_dataframe(
    cm
):
    return pd.DataFrame(
        cm,
        index=[
            "Actual_On-time",
            "Actual_Late"
        ],
        columns=[
            "Predicted_On-time",
            "Predicted_Late"
        ]
    )