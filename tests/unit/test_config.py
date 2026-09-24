from pathlib import Path

from src.config import (
    get_database_config,
    load_config,
    resolve_path,
)


def test_load_config():
    config = load_config()

    assert isinstance(config, dict)
    assert "project" in config
    assert "database" in config
    assert "paths" in config
    assert "features" in config
    assert "model" in config


def test_project_root_is_added():
    config = load_config()

    assert "_project_root" in config
    assert isinstance(
        config["_project_root"],
        Path,
    )


def test_resolve_path():
    config = load_config()

    path = resolve_path(
        config,
        config["paths"]["train"],
    )

    assert isinstance(path, Path)
    assert str(path).endswith(
        "data\\splits\\train.csv"
    ) or str(path).endswith(
        "data/splits/train.csv"
    )


def test_database_configuration():
    config = load_config()

    database = config["database"]

    assert database["user_env"] == "DB_USER"
    assert database["password_env"] == "DB_PASSWORD"
    assert database["host_env"] == "DB_HOST"
    assert database["port_env"] == "DB_PORT"
    assert database["name_env"] == "DB_NAME"


def test_model_configuration():
    config = load_config()

    assert config["model"]["version"]
    assert config["model"]["path"]
    assert config["model"]["preprocessor_path"]