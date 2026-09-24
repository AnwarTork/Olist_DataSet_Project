import pytest
import mlflow

from src.config import load_config
from src.mlflow_tracking import setup_mlflow


@pytest.mark.mlflow
def test_mlflow_connection():

    config = load_config()

    tracking_uri = config["mlflow"]["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()

    client.search_experiments()



# import pytest
# import mlflow
# from mlflow.tracking import MlflowClient

# from src.config import load_config


# @pytest.mark.mlflow
# def test_mlflow_connection():

#     config = load_config()

#     tracking_uri = config["mlflow"]["tracking_uri"]

#     mlflow.set_tracking_uri(tracking_uri)

#     client = MlflowClient()

#     try:
#         client.search_experiments()
#     except Exception as exc:
#         pytest.skip(
#             f"MLflow server is not available at {tracking_uri}: {exc}"
#         )