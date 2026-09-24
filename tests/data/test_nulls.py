import pandas as pd


def test_target_has_no_nulls():
    df = pd.DataFrame(
        {
            "is_late": [0, 1, 0, 1]
        }
    )

    assert not df["is_late"].isna().any()


def test_target_is_binary():
    df = pd.DataFrame(
        {
            "is_late": [0, 1, 0, 1]
        }
    )

    assert set(
        df["is_late"].unique()
    ).issubset({0, 1})