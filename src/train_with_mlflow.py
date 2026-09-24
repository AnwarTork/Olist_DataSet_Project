import joblib

from mlflow.tracking import MlflowClient

from src.config import load_config, resolve_path
from src.mlflow_tracking import (
    setup_mlflow,
    start_run,
    log_parameters,
    log_metrics,
    log_model,
    set_model_alias,
    finish_run,
)
from src.train import (
    train_xgboost,
    predict_model,
    calculate_scale_pos_weight,
)
from src.evaluate import calculate_metrics


def main():

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    config = load_config()

    # ---------------------------------------------------------
    # 2. Setup MLflow
    # ---------------------------------------------------------
    setup_mlflow(config)

    # ---------------------------------------------------------
    # 3. Load processed data
    # ---------------------------------------------------------
    features_path = resolve_path(
        config,
        config["paths"]["features"],
    )

    X_train = joblib.load(
        features_path / "X_train_processed.joblib"
    )

    X_val = joblib.load(
        features_path / "X_val_processed.joblib"
    )

    y_train = joblib.load(
        features_path / "y_train.joblib"
    )

    y_val = joblib.load(
        features_path / "y_val.joblib"
    )

    # ---------------------------------------------------------
    # 4. Calculate class imbalance
    # ---------------------------------------------------------
    scale_pos_weight = calculate_scale_pos_weight(
        y_train
    )

    # ---------------------------------------------------------
    # 5. XGBoost parameters
    # ---------------------------------------------------------
    model_params = {
        "random_state": config["project"]["random_state"],
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }

    # ---------------------------------------------------------
    # 6. Start MLflow run
    # ---------------------------------------------------------
    start_run(
        run_name="xgboost-baseline",
        tags=config["mlflow"]["tags"],
    )

    try:

        # -----------------------------------------------------
        # 7. Log parameters
        # -----------------------------------------------------
        log_parameters(model_params)

        # -----------------------------------------------------
        # 8. Train model
        # -----------------------------------------------------
        model = train_xgboost(
            X_train,
            y_train,
            **model_params,
        )

        # -----------------------------------------------------
        # 9. Validation predictions
        # -----------------------------------------------------
        val_predictions, val_probabilities = predict_model(
            model,
            X_val,
        )

        # -----------------------------------------------------
        # 10. Calculate metrics
        # -----------------------------------------------------
        metrics = calculate_metrics(
            y_val,
            val_predictions,
            val_probabilities,
        )

        print("\nValidation metrics:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # -----------------------------------------------------
        # 11. Log metrics
        # -----------------------------------------------------
        log_metrics(metrics)

        # -----------------------------------------------------
        # 12. Register model
        # -----------------------------------------------------
        registered_model_name = config["mlflow"][
            "registered_model_name"
        ]

        model_info = log_model(
            model=model,
            model_name="xgboost_model",
            input_example=X_val[:5],
            registered_model_name=registered_model_name,
        )

        print("\nModel registered successfully.")

        # -----------------------------------------------------
        # 13. Find registered model version
        # -----------------------------------------------------
        client = MlflowClient()

        versions = client.search_model_versions(
            f"name='{registered_model_name}'"
        )

        latest_version = max(
            int(version.version)
            for version in versions
        )

        # -----------------------------------------------------
        # 14. Assign champion alias
        # -----------------------------------------------------
        alias = config["mlflow"]["model_alias"]

        set_model_alias(
            registered_model_name,
            alias,
            latest_version,
        )

        print(
            f"\nModel version {latest_version} "
            f"assigned to alias '{alias}'."
        )

        print("\nStage 5 training completed successfully.")

    finally:

        # -----------------------------------------------------
        # 15. Finish MLflow run
        # -----------------------------------------------------
        finish_run()


if __name__ == "__main__":
    main()