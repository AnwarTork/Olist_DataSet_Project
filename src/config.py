from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["_project_root"] = PROJECT_ROOT

    return config


def get_database_config(config):
    db_config = config["database"]

    return {
        "user": os.getenv(db_config["user_env"]),
        "password": os.getenv(db_config["password_env"]),
        "host": os.getenv(db_config["host_env"]),
        "port": int(os.getenv(db_config["port_env"], "5432")),
        "name": os.getenv(db_config["name_env"]),
    }


def resolve_path(config, relative_path):
    project_root = config["_project_root"]

    return project_root / relative_path