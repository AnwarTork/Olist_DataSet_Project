import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow import MlflowClient



logger = logging.getLogger(__name__)


def setup_mlflow(config):
    """
    Configure MLflow tracking and experiment.
    """

    mlflow_config = config["mlflow"]

    tracking_uri = mlflow_config["tracking_uri"]
    experiment_name = mlflow_config["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(
        "MLflow configured | tracking_uri=%s | experiment=%s",
        tracking_uri,
        experiment_name,
    )


def start_run(run_name=None, tags=None):
    """
    Start an MLflow run.
    """

    run = mlflow.start_run(run_name=run_name)

    if tags:
        mlflow.set_tags(tags)

    logger.info(
        "MLflow run started | run_id=%s | run_name=%s",
        run.info.run_id,
        run_name,
    )

    return run


def log_parameters(parameters):
    """
    Log model/training parameters.
    """

    clean_parameters = {}

    for key, value in parameters.items():
        if value is None:
            clean_parameters[key] = "None"
        else:
            clean_parameters[key] = value

    mlflow.log_params(clean_parameters)

    logger.info(
        "MLflow parameters logged | count=%d",
        len(clean_parameters),
    )


def log_metrics(metrics):
    """
    Log evaluation metrics.
    """

    clean_metrics = {}

    for key, value in metrics.items():
        if value is not None:
            clean_metrics[key] = float(value)

    mlflow.log_metrics(clean_metrics)

    logger.info(
        "MLflow metrics logged | metrics=%s",
        clean_metrics,
    )


def log_model(
    model,
    model_name,
    input_example=None,
    registered_model_name=None,
):
    signature = None

    if input_example is not None:
        predictions = model.predict(input_example)

        signature = infer_signature(
            input_example,
            predictions,
        )

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_name,
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
    )

    logger.info(
        "MLflow model logged | model_name=%s | registered_name=%s",
        model_name,
        registered_model_name,
    )

    return model_info


def log_artifact(path, artifact_path=None):
    """
    Log a local file as an MLflow artifact.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    mlflow.log_artifact(
        str(path),
        artifact_path=artifact_path,
    )

    logger.info(
        "MLflow artifact logged | path=%s",
        path,
    )


def set_model_alias(
    registered_model_name,
    alias,
    version,
):
    """
    Assign an alias such as 'champion'
    to a registered model version.
    """

    client = MlflowClient()

    client.set_registered_model_alias(
        registered_model_name,
        alias,
        version,
    )

    logger.info(
        "Model alias assigned | model=%s | alias=%s | version=%s",
        registered_model_name,
        alias,
        version,
    )


def get_model_version(
    registered_model_name,
    version,
):
    """
    Retrieve registered model version information.
    """

    client = MlflowClient()

    return client.get_model_version(
        registered_model_name,
        version,
    )


def get_model_by_alias(
    registered_model_name,
    alias,
):
    """
    Retrieve the model version currently assigned
    to an alias.
    """

    client = MlflowClient()

    return client.get_model_version_by_alias(
        registered_model_name,
        alias,
    )


def load_registered_model(
    registered_model_name,
    alias,
):
    """
    Load a model from MLflow Model Registry.
    """

    model_uri = (
        f"models:/{registered_model_name}@{alias}"
    )

    logger.info(
        "Loading model from registry | uri=%s",
        model_uri,
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    return model


def finish_run():
    """
    End the active MLflow run.
    """

    active_run = mlflow.active_run()

    if active_run:
        logger.info(
            "MLflow run finished | run_id=%s",
            active_run.info.run_id,
        )

    mlflow.end_run()