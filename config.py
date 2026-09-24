from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    config_path = PROJECT_ROOT / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


CONFIG = load_config()


DATA_RAW_DIR = PROJECT_ROOT / CONFIG["paths"]["data_raw"]
DATA_PROCESSED_DIR = PROJECT_ROOT / CONFIG["paths"]["data_processed"]
DATA_SPLITS_DIR = PROJECT_ROOT / CONFIG["paths"]["data_splits"]
MODELS_DIR = PROJECT_ROOT / CONFIG["paths"]["models"]


FEATURES_FILE = DATA_PROCESSED_DIR / CONFIG["files"]["features"]
LABELED_FILE = DATA_PROCESSED_DIR / CONFIG["files"]["labeled"]

TRAIN_FILE = DATA_SPLITS_DIR / CONFIG["files"]["train"]
VALIDATION_FILE = DATA_SPLITS_DIR / CONFIG["files"]["validation"]
TEST_FILE = DATA_SPLITS_DIR / CONFIG["files"]["test"]