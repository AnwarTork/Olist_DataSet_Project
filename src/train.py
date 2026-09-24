import joblib
import pandas as pd

from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


def train_logistic_regression(
    X_train,
    y_train,
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
):
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_model(
    model,
    X,
):
    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    return predictions, probabilities


def tune_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    C_values,
    class_weights,
    max_iter=1000,
    random_state=42,
):
    results = []

    for C, class_weight in product(
        C_values,
        class_weights,
    ):
        model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )

        model.fit(
            X_train,
            y_train,
        )

        val_pred = model.predict(X_val)

        val_proba = model.predict_proba(
            X_val
        )[:, 1]

        precision = precision_score(
            y_val,
            val_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            val_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            val_pred,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_val,
            val_proba,
        )

        pr_auc = average_precision_score(
            y_val,
            val_proba,
        )

        results.append(
            {
                "C": C,
                "class_weight": class_weight,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = (
        results_df
        .sort_values(
            by="f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_result = results_df.iloc[0]

    best_C = best_result["C"]

    best_class_weight = (
        best_result["class_weight"]
    )

    best_model = LogisticRegression(
        C=best_C,
        class_weight=best_class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    best_model.fit(
        X_train,
        y_train,
    )

    return (
        results_df,
        best_model,
        best_C,
        best_class_weight,
    )


def train_random_forest(
    X_train,
    y_train,
    random_state=42,
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def calculate_scale_pos_weight(y_train):
    negative = (
        y_train == 0
    ).sum()

    positive = (
        y_train == 1
    ).sum()

    if positive == 0:
        raise ValueError(
            "No positive samples found."
        )

    return negative / positive


def train_xgboost(
    X_train,
    y_train,
    random_state=42,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=None,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
):
    if scale_pos_weight is None:
        scale_pos_weight = (
            calculate_scale_pos_weight(
                y_train
            )
        )

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        objective=objective,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def save_model(
    model,
    output_path,
):
    output_path = str(output_path)

    joblib.dump(
        model,
        output_path,
    )


def load_model(
    model_path,
):
    return joblib.load(
        model_path
    )