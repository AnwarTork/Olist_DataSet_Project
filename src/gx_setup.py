from pathlib import Path

import great_expectations as gx


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_gx_context():
    """
    Return the persistent Great Expectations
    File Data Context for this project.
    """

    gx_root = PROJECT_ROOT / "great_expectations"

    gx_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    context = gx.get_context(
        mode="file",
        project_root_dir=str(PROJECT_ROOT),
    )

    return context